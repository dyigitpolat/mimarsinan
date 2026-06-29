"""Harvest the MNIST ``mlp_mixer_core`` closure ledger from LOCAL phased-deployment dirs.

The 30 ``generated/<run>_phased_deployment_run`` dirs lack the ``stdout.log``/``exitcode``
files the generic Slurmech :mod:`artifact_classifier` needs. This reads each run's
``_GUI_STATE/run_info.json`` (``status`` -> returncode proxy, ``finished_at-started_at``
-> wall) and ``__target_metric.json`` (deployed_acc), recovers hypervolume axes from the
queue manifest (by exact id, else by ``cell_id``), and emits a schema-consistent v1 ledger
(via the shared :class:`ArtifactRecord` normalization) plus a per-``(cell_id, recipe_id)``
family rollup. No GPU, no job submission.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from artifact_classifier import ArtifactRecord, _STATUS_OWNERS, _load_target_metric

from mimarsinan.chip_simulation.ledger_schema import fastest_successful_baseline_wall_s
from mimarsinan.chip_simulation.parity_contract import parity_contract_metadata

RUN_PREFIX = "mnist_mixer_diag_"
RUN_SUFFIX = "_phased_deployment_run"
STUDY = "MNIST_MIXER_DIAGNOSTICS"
CLUSTER = "MNIST_MIXER_CLOSURE"

DEFAULT_MIN_ACC = 0.97
# deployed acc below 2x MNIST chance (0.10) is a genuine recipe collapse (MEASURED_DEAD),
# not a healthy-accuracy model rejected by a parity/faithfulness gate.
COLLAPSE_ACC = 0.20
# Operative baseline for the v2 slurmech H100 wave. Recorded for traceability only; the v1
# rows' relative_time is vs the LOCAL analytical-control wall (CPU runs) -- never divide a
# local CPU wall by this H100 figure.
CANONICAL_SLURMECH_BASELINE_S = 428.0

CONTROLLER_RECIPE_ID = "mixer_controller_baseline"
CONTROLLER_FAMILY = "controller"
ANALYTICAL_CONTROL_ROLE = "analytical_control"

CELL_LEVEL_FIELDS = (
    "study",
    "cluster",
    "diagnostic_role",
    "vehicle",
    "dataset",
    "firing",
    "sync",
    "backend",
    "acceptance_min_deployed_acc",
    "acceptance_max_relative_time",
)

DEAD = "MEASURED_DEAD"
PASS = "VALID_97_FAST"
FLAG = "VALID_FLAGGED_WITH_OWNER"
INCOMPLETE = "INCOMPLETE"
# A recipe that TRAINS to healthy accuracy but is rejected by a fidelity/parity gate
# (rc=1, no collapsed seed). Honestly distinct from MEASURED_DEAD (genuine accuracy
# collapse) -- loosening a budget would "fix" the former but never the latter.
GATE_REJECTED = "GATE_REJECTED"


# --------------------------------------------------------------------------- manifest axes


@dataclass(frozen=True)
class ManifestIndex:
    """Axis lookup over the queue manifest, exact-by-id with a cell_id fallback."""

    by_id: dict[str, dict[str, Any]]
    cell_axes: dict[str, dict[str, Any]]
    cell_diag_family: dict[str, str]
    cell_ids: tuple[str, ...]


def build_manifest_index(manifest: Iterable[Mapping[str, Any]]) -> ManifestIndex:
    by_id: dict[str, dict[str, Any]] = {}
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in manifest:
        tags = dict(job["tags"])
        by_id[str(job["id"])] = tags
        by_cell[str(tags["cell_id"])].append(tags)
    cell_axes: dict[str, dict[str, Any]] = {}
    cell_diag_family: dict[str, str] = {}
    for cell, tag_list in by_cell.items():
        axes = {field: tag_list[0].get(field) for field in CELL_LEVEL_FIELDS}
        axes["cell_id"] = cell
        cell_axes[cell] = axes
        families = sorted({t["recipe_family"] for t in tag_list if t["recipe_family"] != CONTROLLER_FAMILY})
        cell_diag_family[cell] = families[0] if families else CONTROLLER_FAMILY
    cell_ids = tuple(sorted(by_cell, key=len, reverse=True))
    return ManifestIndex(by_id, cell_axes, cell_diag_family, cell_ids)


def parse_run_id(run_id: str, cell_ids: Iterable[str]) -> tuple[str, str, int]:
    """Split ``run_id`` into ``(cell_id, recipe_id, seed)`` via longest cell_id prefix."""
    body = run_id[len(RUN_PREFIX):] if run_id.startswith(RUN_PREFIX) else run_id
    for cell in cell_ids:
        prefix = cell + "_"
        if body.startswith(prefix):
            rest = body[len(prefix):]
            match = re.fullmatch(r"(?P<recipe>.+)_s(?P<seed>\d+)", rest)
            if match:
                return cell, match.group("recipe"), int(match.group("seed"))
            break
    raise ValueError(f"cannot parse run_id against manifest cells: {run_id}")


def recover_axes(run_id: str, index: ManifestIndex) -> tuple[dict[str, Any], str]:
    """Return ``(axes, axes_source)``; exact manifest tags when present, else cell-derived."""
    cell_id, recipe_id, seed = parse_run_id(run_id, index.cell_ids)
    if run_id in index.by_id:
        axes = dict(index.by_id[run_id])
        axes["seed"] = seed
        return axes, "manifest_exact"
    axes = dict(index.cell_axes[cell_id])
    axes["recipe_id"] = recipe_id
    axes["recipe_family"] = (
        CONTROLLER_FAMILY if recipe_id == CONTROLLER_RECIPE_ID else index.cell_diag_family[cell_id]
    )
    axes["batch_id"] = f"{RUN_PREFIX}{cell_id}_{recipe_id}"
    axes["budget_schedule"] = None
    axes["required_probes"] = []
    axes["seed"] = seed
    return axes, "manifest_cell_derived"


# --------------------------------------------------------------------------- local run read


COST_RECORD_NAME = "cost_record.json"
# The HONEST deployment-efficiency axes (on-chip), distinct from pipeline wall: a
# genuine-spiking recipe's one-time QAT training makes its wall longer than the
# analytical control's, but its DEPLOYED latency/energy is what actually competes.
COST_FIELDS = ("latency_steps", "mj_per_sample", "spikes", "energy_proxy_neuron_steps",
               "cores", "s_global", "acc_deploy")


@dataclass(frozen=True)
class DeployCost:
    """On-chip deployment cost from a run's ``cost_record.json`` (sim-measured)."""

    latency_steps: int | None
    mj_per_sample: float | None
    spikes: int | None
    energy_proxy_neuron_steps: int | None
    cores: int | None
    s_global: int | None
    acc_deploy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in COST_FIELDS}


def read_cost_record(run_dir: Path) -> DeployCost | None:
    """Parse ``cost_record.json`` (absent for crashed runs that never reached sim)."""
    path = run_dir / COST_RECORD_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(errors="ignore"))
    except json.JSONDecodeError:
        return None
    return DeployCost(**{field: payload.get(field) for field in COST_FIELDS})


def read_run_info(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "_GUI_STATE" / "run_info.json").read_text(errors="ignore"))


def local_wall_s(info: Mapping[str, Any]) -> float | None:
    started, finished = info.get("started_at"), info.get("finished_at")
    if started is None or finished is None:
        return None
    return float(finished) - float(started)


def returncode_for_status(status: str) -> int | None:
    return {"completed": 0, "failed": 1}.get(status)


def _clean_error(error: Any) -> str | None:
    if not error:
        return None
    return re.sub(r"\s+", " ", str(error)).strip()


def _gate_completed(acc: float | None, rel: float | None, min_acc: float) -> tuple[str, str | None]:
    if acc is None:
        return FLAG, "missing_deployed_metric"
    if acc < min_acc:
        return FLAG, "accuracy_below_97"
    if rel is not None and rel >= 1.0:
        return FLAG, "slower_than_baseline"
    return PASS, None


def _gate_failed(acc: float | None, error: str | None, collapse_acc: float) -> tuple[str, str | None]:
    text = (error or "").lower()
    collapsed = ("failed to retain performance" in text) or (acc is not None and acc < collapse_acc)
    if collapsed:
        return DEAD, "measured_dead"
    if acc is None:
        return FLAG, "returncode_nonzero"
    return FLAG, "parity_failure"


@dataclass(frozen=True)
class LocalRun:
    run_id: str
    run_dir: Path
    status: str
    returncode: int | None
    deployed_acc: float | None
    wall_s: float | None
    error: str | None
    axes: dict[str, Any]
    axes_source: str
    cost: DeployCost | None = None

    @property
    def incomplete(self) -> bool:
        return self.returncode is None or self.wall_s is None

    @property
    def is_analytical_control(self) -> bool:
        return self.axes.get("diagnostic_role") == ANALYTICAL_CONTROL_ROLE


def read_local_run(run_dir: Path, index: ManifestIndex) -> LocalRun:
    run_id = run_dir.name[: -len(RUN_SUFFIX)] if run_dir.name.endswith(RUN_SUFFIX) else run_dir.name
    info = read_run_info(run_dir)
    status = str(info.get("status") or "")
    axes, axes_source = recover_axes(run_id, index)
    return LocalRun(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        returncode=returncode_for_status(status),
        deployed_acc=_load_target_metric(run_dir),
        wall_s=local_wall_s(info),
        error=_clean_error(info.get("error")),
        axes=axes,
        axes_source=axes_source,
        cost=read_cost_record(run_dir),
    )


def _fastest_analytical_winner(runs: Iterable[LocalRun], min_acc: float) -> LocalRun | None:
    """The fastest-wall analytical-control success — the canonical baseline run."""
    eligible = [
        run
        for run in runs
        if run.is_analytical_control and run.returncode == 0
        and run.deployed_acc is not None and run.deployed_acc >= min_acc
        and run.wall_s is not None
    ]
    return min(eligible, key=lambda r: r.wall_s) if eligible else None


def local_baseline(runs: Iterable[LocalRun], min_acc: float) -> tuple[float | None, str | None]:
    """Fastest analytical-control success wall (the LOCAL relative-time reference)."""
    runs = list(runs)
    winner = _fastest_analytical_winner(runs, min_acc)
    if winner is None:
        return None, None
    shaped = [{"returncode": 0, "deployed_acc": winner.deployed_acc, "wall_s": winner.wall_s}]
    return fastest_successful_baseline_wall_s(shaped), winner.run_id


def cost_baseline(runs: Iterable[LocalRun], min_acc: float) -> DeployCost | None:
    """On-chip deploy cost of the canonical analytical-control baseline run."""
    winner = _fastest_analytical_winner(list(runs), min_acc)
    return winner.cost if winner is not None else None


# --------------------------------------------------------------------------- ledger rows


def classify_local_run(run: LocalRun, *, baseline_wall_s: float | None, min_acc: float) -> ArtifactRecord:
    rel = None
    if baseline_wall_s and run.wall_s is not None:
        rel = run.wall_s / baseline_wall_s
    parity = parity_contract_metadata(run.axes) if run.axes else {}
    if run.incomplete:
        return ArtifactRecord(
            artifact_dir=str(run.run_dir),
            name=run.run_id,
            returncode=None,
            deployed_acc=None,
            final_step=None,
            profiled_wall_s=0.0,
            relative_time=None,
            status=INCOMPLETE,
            failure_reason="incomplete_run",
            owner="execution",
            repair_path="Re-run; orphaned (status=running, no finished_at).",
            parity_contracts=parity,
        )
    if run.returncode == 0:
        status, reason = _gate_completed(run.deployed_acc, rel, min_acc)
    else:
        status, reason = _gate_failed(run.deployed_acc, run.error, COLLAPSE_ACC)
    owner, repair = (None, None) if reason is None else _STATUS_OWNERS.get(reason, ("unknown", "investigate"))
    return ArtifactRecord(
        artifact_dir=str(run.run_dir),
        name=run.run_id,
        returncode=run.returncode,
        deployed_acc=run.deployed_acc,
        final_step=None,
        profiled_wall_s=run.wall_s,
        relative_time=rel,
        status=status,
        failure_reason=reason,
        owner=owner,
        repair_path=repair,
        parity_contracts=parity,
    )


def _relative_deploy_cost(value: float | None, baseline: float | None) -> float | None:
    if value is None or not baseline:
        return None
    return value / baseline


def build_row(
    run: LocalRun, record: ArtifactRecord, *, baseline_wall_s: float | None,
    baseline_run_id: str | None, baseline_cost: DeployCost | None = None,
) -> dict[str, Any]:
    row = record.to_ledger_row(study=STUDY, cluster=CLUSTER, axes=run.axes)
    row["seed"] = run.axes.get("seed")
    row["axes_source"] = run.axes_source
    row["failure_detail"] = run.error
    row["timing_baseline_wall_s"] = baseline_wall_s
    row["timing_baseline_kind"] = "local_fastest_analytical_control_success"
    row["timing_baseline_run_id"] = baseline_run_id
    row["canonical_slurmech_baseline_wall_s"] = CANONICAL_SLURMECH_BASELINE_S
    row["local_relative_time"] = row.get("relative_time")
    row["faster_than_baseline"] = (
        None if row.get("relative_time") is None else row["relative_time"] < 1.0
    )
    row["deploy_cost"] = run.cost.to_dict() if run.cost is not None else None
    base_lat = baseline_cost.latency_steps if baseline_cost else None
    base_energy = baseline_cost.mj_per_sample if baseline_cost else None
    run_lat = run.cost.latency_steps if run.cost else None
    run_energy = run.cost.mj_per_sample if run.cost else None
    row["relative_deploy_latency"] = _relative_deploy_cost(run_lat, base_lat)
    row["relative_deploy_energy"] = _relative_deploy_cost(run_energy, base_energy)
    provenance = dict(row.get("provenance") or {})
    provenance["run_dir"] = str(run.run_dir)
    row["provenance"] = provenance
    if record.status == INCOMPLETE:
        row["returncode"] = None
        row["run_wall_s"] = None
        row["relative_time"] = None
        row["local_relative_time"] = None
        row["faster_than_baseline"] = None
        row["incomplete"] = True
        row["observed_partial_metric"] = run.deployed_acc
    return row


def harvest_rows(generated_root: Path, index: ManifestIndex, *, min_acc: float = DEFAULT_MIN_ACC):
    """Return ``(rows, meta)`` for every ``*_phased_deployment_run`` dir under ``generated_root``."""
    run_dirs = sorted(p for p in generated_root.glob(f"{RUN_PREFIX}*{RUN_SUFFIX}") if p.is_dir())
    runs = [read_local_run(run_dir, index) for run_dir in run_dirs]
    baseline_wall_s, baseline_run_id = local_baseline(runs, min_acc)
    baseline_cost = cost_baseline(runs, min_acc)
    rows = []
    for run in runs:
        record = classify_local_run(run, baseline_wall_s=baseline_wall_s, min_acc=min_acc)
        rows.append(build_row(run, record, baseline_wall_s=baseline_wall_s,
                              baseline_run_id=baseline_run_id, baseline_cost=baseline_cost))
    rows.sort(key=lambda r: r["run_id"])
    meta = {
        "n_rows": len(rows),
        "n_incomplete": sum(1 for r in rows if r.get("incomplete")),
        "baselines": {
            "local_fastest_analytical_control_success_wall_s": baseline_wall_s,
            "local_baseline_run_id": baseline_run_id,
            "canonical_slurmech_analytical_baseline_s": CANONICAL_SLURMECH_BASELINE_S,
            "deploy_cost": baseline_cost.to_dict() if baseline_cost is not None else None,
            "note": (
                "v1 row relative_time is vs the LOCAL analytical-control success WALL (CPU runs). "
                "The canonical slurmech analytical baseline is 428.0s and is the operative baseline "
                "for the v2 slurmech H100 wave; do NOT divide these local CPU walls by 428.0s. "
                "relative_deploy_latency/energy are the HONEST on-chip efficiency axes (vs the "
                "analytical control's cost_record), unaffected by one-time QAT training wall."
            ),
        },
        "acceptance_gate": {
            "min_deployed_acc": min_acc,
            "max_relative_time": 1.0,
            "require_returncode": 0,
        },
        "collapse_acc_threshold": COLLAPSE_ACC,
    }
    return rows, meta


# --------------------------------------------------------------------------- family rollup


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def classify_family(rows: list[Mapping[str, Any]]) -> str:
    if any(r.get("deployment_validity") == DEAD for r in rows):
        return DEAD
    completed = [r for r in rows if r.get("returncode") == 0]
    rejected = [r for r in rows if r.get("returncode") == 1]
    if rejected:
        return GATE_REJECTED
    if not completed:
        return INCOMPLETE
    if all(r["deployment_validity"] == PASS for r in completed):
        return PASS
    return "VALID_FLAGGED"


def build_rollup(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        axes = row.get("axes") or {}
        groups[(axes.get("cell_id"), axes.get("recipe_id"))].append(row)
    families = []
    for (cell_id, recipe_id), group in sorted(groups.items()):
        accs = [r["deployed_acc"] for r in group if r.get("deployed_acc") is not None]
        walls = [r["run_wall_s"] for r in group if r.get("run_wall_s") is not None]
        rels = [r["local_relative_time"] for r in group if r.get("local_relative_time") is not None]
        latencies = [r["deploy_cost"]["latency_steps"] for r in group
                     if r.get("deploy_cost") and r["deploy_cost"].get("latency_steps") is not None]
        energies = [r["deploy_cost"]["mj_per_sample"] for r in group
                    if r.get("deploy_cost") and r["deploy_cost"].get("mj_per_sample") is not None]
        rel_lat = [r["relative_deploy_latency"] for r in group if r.get("relative_deploy_latency") is not None]
        rel_energy = [r["relative_deploy_energy"] for r in group if r.get("relative_deploy_energy") is not None]
        axes0 = group[0].get("axes") or {}
        families.append(
            {
                "cell_id": cell_id,
                "recipe_id": recipe_id,
                "recipe_family": axes0.get("recipe_family"),
                "firing": axes0.get("firing"),
                "sync": axes0.get("sync"),
                "classification": classify_family(group),
                "seeds": sorted(r.get("seed") for r in group if r.get("seed") is not None),
                "n_seeds": len(group),
                "n_pass": sum(1 for r in group if r["deployment_validity"] == PASS),
                "n_flag": sum(1 for r in group if r["deployment_validity"] == FLAG),
                "n_dead": sum(1 for r in group if r["deployment_validity"] == DEAD),
                "n_incomplete": sum(1 for r in group if r["deployment_validity"] == INCOMPLETE),
                "acc": _stats(accs),
                "wall_s": _stats(walls),
                "relative_time": _stats(rels),
                "latency_steps": _stats(latencies),
                "mj_per_sample": _stats(energies),
                "relative_deploy_latency": _stats(rel_lat),
                "relative_deploy_energy": _stats(rel_energy),
                "failure_reasons": sorted({r["failure_reason"] for r in group if r.get("failure_reason")}),
                "owners": sorted({r["owner"] for r in group if r.get("owner")}),
                "axes_sources": sorted({r.get("axes_source") for r in group if r.get("axes_source")}),
            }
        )
    return families


def rollup_counts(families: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for family in families:
        counts[family["classification"]] += 1
    return dict(counts)


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def render_rollup_table(families: Iterable[Mapping[str, Any]]) -> str:
    header = (
        f"{'cell_id':38s} {'recipe_id':40s} {'seeds':7s} "
        f"{'acc(min..max)':16s} {'wall_s(min..max)':18s} {'classification':14s}"
    )
    lines = [header, "-" * len(header)]
    for family in families:
        acc = family["acc"]
        wall = family["wall_s"]
        seeds = ",".join(str(s) for s in family["seeds"]) or "-"
        acc_range = f"{_fmt(acc['min'], '.3f')}..{_fmt(acc['max'], '.3f')}"
        wall_range = f"{_fmt(wall['min'], '.0f')}..{_fmt(wall['max'], '.0f')}"
        lines.append(
            f"{family['cell_id']:38s} {family['recipe_id']:40s} {seeds:7s} "
            f"{acc_range:16s} {wall_range:18s} {family['classification']:14s}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", default="generated")
    parser.add_argument("--manifest", default="runs/campaign/mnist_mixer_queue_manifest.json")
    parser.add_argument("--ledger-out", default="runs/campaign/mnist_mixer_ledger_v1.json")
    parser.add_argument("--rollup-out", default="runs/campaign/mnist_mixer_ledger_v1_rollup.json")
    parser.add_argument("--min-deployed-acc", type=float, default=DEFAULT_MIN_ACC)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    index = build_manifest_index(manifest)
    rows, meta = harvest_rows(Path(args.generated_root), index, min_acc=args.min_deployed_acc)
    families = build_rollup(rows)
    counts = rollup_counts(families)

    Path(args.ledger_out).write_text(json.dumps(rows, indent=2) + "\n")
    rollup = {"header": {**meta, "manifest": args.manifest, "generated_root": args.generated_root},
              "families": families,
              "family_classification_counts": counts}
    Path(args.rollup_out).write_text(json.dumps(rollup, indent=2) + "\n")

    print(render_rollup_table(families))
    print()
    print(f"family classification counts: {counts}")
    print(f"rows={meta['n_rows']} incomplete={meta['n_incomplete']} "
          f"local_baseline_wall_s={meta['baselines']['local_fastest_analytical_control_success_wall_s']}")
    print(f"ledger -> {args.ledger_out}")
    print(f"rollup -> {args.rollup_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
