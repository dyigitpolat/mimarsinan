"""Generic Slurmech / run artifact classifier for hypervolume closure."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from mimarsinan.chip_simulation.ledger_schema import (
    fastest_successful_baseline_wall_s,
    normalize_ledger_record,
    with_relative_timing,
)
from mimarsinan.chip_simulation.parity_contract import parity_contract_metadata

_PROFILE_RE = re.compile(
    r"\[PROFILE\] step='(?P<step>[^']+)' wall=\s*(?P<wall>[0-9.]+)s metric=(?P<metric>[0-9.]+)"
)

_STATUS_OWNERS: dict[str, tuple[str, str]] = {
    "returncode_nonzero": ("execution", "Fix rc=0 substrate or recipe crash"),
    "missing_exitcode": ("execution", "Ensure Slurmech child exitcode capture"),
    "missing_deployed_metric": ("measurement", "Require Simulation/__target_metric.json"),
    "accuracy_below_97": ("conversion", "Repair recipe accuracy or flag cell"),
    "slower_than_baseline": ("tuning", "Reduce tuning wall or promote faster recipe"),
    "measured_dead": ("research", "Keep negative result; do not promote recipe"),
    "parity_failure": ("faithfulness", "Fix NF/SCM or torch/deployed parity gate"),
}


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_dir: str
    name: str
    returncode: int | None
    deployed_acc: float | None
    final_step: str | None
    profiled_wall_s: float
    relative_time: float | None
    status: str
    failure_reason: str | None
    owner: str | None = None
    repair_path: str | None = None
    parity_contracts: dict[str, str] = field(default_factory=dict)
    step_metrics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_ledger_row(self, *, study: str, cluster: str, axes: Mapping[str, Any]) -> dict[str, Any]:
        axis_payload = dict(axes)
        row = {
            "row_type": "measured",
            "study": study,
            "cluster": cluster,
            "run_id": self.name,
            "model": axis_payload.get("vehicle") or axis_payload.get("model"),
            "dataset": axis_payload.get("dataset"),
            "spiking_mode": axis_payload.get("firing") or axis_payload.get("spiking_mode"),
            "returncode": self.returncode if self.returncode is not None else 1,
            "deployed_acc": self.deployed_acc,
            "run_wall_s": self.profiled_wall_s,
            "relative_time": self.relative_time,
            "deployment_validity": self.status,
            "failure_reason": self.failure_reason,
            "owner": self.owner,
            "repair_path": self.repair_path,
            "parity_contracts": dict(self.parity_contracts),
            "step_metrics": list(self.step_metrics),
            "axes": axis_payload,
        }
        return normalize_ledger_record(row, require_science=False)


def iter_artifact_dirs(root: str | Path) -> Iterable[Path]:
    """Yield single-run artifact dirs or pack child dirs under ``root``."""
    root = Path(root)
    children = root / "children"
    if children.is_dir():
        yield from sorted(path for path in children.iterdir() if path.is_dir())
        return
    if (root / "stdout.log").exists() or (root / "exitcode").exists():
        yield root


def _read_exitcode(path: Path) -> int | None:
    p = path / "exitcode"
    if not p.exists():
        return None
    text = p.read_text(errors="ignore").strip()
    try:
        return int(text)
    except ValueError:
        return None


def _profiles(stdout: str) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for match in _PROFILE_RE.finditer(stdout):
        out.append((
            match.group("step"),
            float(match.group("wall")),
            float(match.group("metric")),
        ))
    return out


def _load_target_metric(artifact_dir: Path, generated_root: Path | None = None) -> float | None:
    candidates: list[Path] = []
    for name in ("__target_metric.json", "generated/__target_metric.json"):
        candidates.append(artifact_dir / name)
    if generated_root is not None:
        candidates.append(generated_root / "__target_metric.json")
    for path in candidates:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(errors="ignore"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, (int, float)):
                return float(payload)
            if isinstance(payload, dict):
                for key in ("deployed_acc", "accuracy", "value"):
                    if key in payload and payload[key] is not None:
                        return float(payload[key])
    return None


def _find_generated_run_dir(artifact_dir: Path) -> Path | None:
    for path in sorted(artifact_dir.glob("generated/*_deployment_run")):
        if path.is_dir():
            return path
    nested = artifact_dir / "generated"
    if nested.is_dir():
        for path in sorted(nested.glob("*_deployment_run")):
            if path.is_dir():
                return path
    return None


def _load_step_metrics(artifact_dir: Path) -> list[dict[str, Any]]:
    run_dir = _find_generated_run_dir(artifact_dir)
    if run_dir is None:
        return []
    steps_path = run_dir / "_GUI_STATE" / "steps.json"
    if not steps_path.is_file():
        return []
    try:
        payload = json.loads(steps_path.read_text(errors="ignore"))
    except json.JSONDecodeError:
        return []
    steps = (payload or {}).get("steps") or {}
    out: list[dict[str, Any]] = []
    for name, body in steps.items():
        if not isinstance(body, dict):
            continue
        out.append(
            {
                "name": str(name),
                "status": "measured",
                "metrics": dict(body.get("metrics") or {}),
                "timing": {"wall_s": body.get("wall_s"), "relative_to_baseline": None},
            }
        )
    return out


def classify_artifact(
    artifact_dir: str | Path,
    *,
    fastest_baseline_wall_s: float | None,
    min_deployed_acc: float = 0.97,
    axes: Mapping[str, Any] | None = None,
) -> ArtifactRecord:
    """Classify one fetched Slurmech artifact directory."""
    artifact_dir = Path(artifact_dir)
    stdout = (artifact_dir / "stdout.log").read_text(errors="ignore") if (artifact_dir / "stdout.log").exists() else ""
    stderr = (artifact_dir / "stderr.log").read_text(errors="ignore") if (artifact_dir / "stderr.log").exists() else ""
    profiles = _profiles(stdout)
    final_step = profiles[-1][0] if profiles else None
    deployed_acc = profiles[-1][2] if profiles else None
    generated_run = _find_generated_run_dir(artifact_dir)
    metric_from_file = _load_target_metric(artifact_dir, generated_run)
    if metric_from_file is not None:
        deployed_acc = metric_from_file
    wall = sum(item[1] for item in profiles)
    relative_time = None
    if fastest_baseline_wall_s and fastest_baseline_wall_s > 0.0:
        relative_time = wall / float(fastest_baseline_wall_s)
    rc = _read_exitcode(artifact_dir)
    step_metrics = _load_step_metrics(artifact_dir)
    axis_row = dict(axes or {})
    parity_contracts = parity_contract_metadata(axis_row) if axis_row else {}

    status = "VALID_97_FAST"
    failure_reason = None
    if "MEASURED_DEAD" in stderr:
        status = "MEASURED_DEAD"
        failure_reason = "measured_dead"
    elif "parity" in stderr.lower() and "fail" in stderr.lower():
        status = "VALID_FLAGGED_WITH_OWNER"
        failure_reason = "parity_failure"
    elif rc != 0:
        status = "VALID_FLAGGED_WITH_OWNER"
        failure_reason = "returncode_nonzero" if rc is not None else "missing_exitcode"
    elif deployed_acc is None:
        status = "VALID_FLAGGED_WITH_OWNER"
        failure_reason = "missing_deployed_metric"
    elif deployed_acc < min_deployed_acc:
        status = "VALID_FLAGGED_WITH_OWNER"
        failure_reason = "accuracy_below_97"
    elif relative_time is not None and relative_time >= 1.0:
        status = "VALID_FLAGGED_WITH_OWNER"
        failure_reason = "slower_than_baseline"

    owner = None
    repair_path = None
    if failure_reason is not None:
        owner, repair_path = _STATUS_OWNERS.get(failure_reason, ("unknown", "investigate"))

    return ArtifactRecord(
        artifact_dir=str(artifact_dir),
        name=artifact_dir.name,
        returncode=rc,
        deployed_acc=deployed_acc,
        final_step=final_step,
        profiled_wall_s=wall,
        relative_time=relative_time,
        status=status,
        failure_reason=failure_reason,
        owner=owner,
        repair_path=repair_path,
        parity_contracts=parity_contracts,
        step_metrics=step_metrics,
    )


def classify_many(
    roots: Iterable[str | Path],
    *,
    fastest_baseline_wall_s: float | None,
    min_deployed_acc: float = 0.97,
) -> list[ArtifactRecord]:
    records = []
    for root in roots:
        for artifact_dir in iter_artifact_dirs(root):
            records.append(
                classify_artifact(
                    artifact_dir,
                    fastest_baseline_wall_s=fastest_baseline_wall_s,
                    min_deployed_acc=min_deployed_acc,
                )
            )
    return records


def classify_to_ledger_rows(
    records: Iterable[ArtifactRecord],
    *,
    study: str,
    cluster: str,
    axes_by_name: Mapping[str, Mapping[str, Any]] | None = None,
    baseline_from_successes: bool = False,
) -> list[dict[str, Any]]:
    record_list = list(records)
    baseline = None
    if baseline_from_successes:
        successes = [
            {
                "returncode": r.returncode or 0,
                "deployed_acc": r.deployed_acc,
                "wall_s": r.profiled_wall_s,
            }
            for r in record_list
            if r.status == "VALID_97_FAST"
        ]
        if successes:
            baseline = fastest_successful_baseline_wall_s(successes)
    rows = []
    for record in record_list:
        axes = dict((axes_by_name or {}).get(record.name, {}))
        row = record.to_ledger_row(study=study, cluster=cluster, axes=axes)
        if baseline is not None and record.profiled_wall_s:
            row = with_relative_timing(row, baseline)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--fastest-baseline-wall-s", type=float, default=None)
    parser.add_argument("--min-deployed-acc", type=float, default=0.97)
    parser.add_argument("--study", default="HYPERVOLUME_CLOSURE")
    parser.add_argument("--cluster", default="ARTIFACT_CLASSIFIER")
    parser.add_argument("--ledger-out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    records = classify_many(
        args.artifacts,
        fastest_baseline_wall_s=args.fastest_baseline_wall_s,
        min_deployed_acc=args.min_deployed_acc,
    )
    if args.ledger_out:
        rows = classify_to_ledger_rows(
            records,
            study=args.study,
            cluster=args.cluster,
            baseline_from_successes=args.fastest_baseline_wall_s is None,
        )
        Path(args.ledger_out).write_text(json.dumps(rows, indent=2) + "\n")
    if args.json:
        print(json.dumps([record.to_dict() for record in records], indent=2))
    else:
        for record in records:
            print(
                f"{record.name}: {record.status} acc={record.deployed_acc} "
                f"rel_time={record.relative_time} owner={record.owner} "
                f"reason={record.failure_reason}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
