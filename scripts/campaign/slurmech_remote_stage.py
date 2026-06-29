"""Stage remote slurmech-pack phased-deployment artifacts into a LOCAL generated/ tree.

``slurmech fetch`` only pulls ``stdout.log``/``stderr.log``/``exitcode`` — never the
``generated/<run>_phased_deployment_run`` outputs the ledger harvester needs. This
reuses slurmech's SSH ``conn`` to pull just the two small JSONs per run
(``__target_metric.json`` + ``_GUI_STATE/run_info.json``) into a staging tree shaped
exactly like a LOCAL run dir, so :mod:`mnist_mixer_local_harvest` can harvest it
verbatim — no remote re-launch, no row-logic duplication.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mnist_mixer_local_harvest import RUN_SUFFIX  # noqa: E402

TARGET_METRIC_NAME = "__target_metric.json"
RUN_INFO_REL = os.path.join("_GUI_STATE", "run_info.json")
COST_RECORD_NAME = "cost_record.json"
GENERATED_SUBDIR = "workspace/generated"

# Only runs that reach the SANA-FE sim (rc=0) emit cost_record.json; a crashed
# rc=1 run dies at an earlier gate, so the on-chip cost is genuinely absent.
OPTIONAL_ARTIFACTS = ("cost_record",)


def phased_run_dirname(run_name: str) -> str:
    """The ``generated/`` subdir for a child run name (idempotent on the suffix)."""
    if run_name.endswith(RUN_SUFFIX):
        return run_name
    return f"{run_name}{RUN_SUFFIX}"


def run_name_from_dirname(dirname: str) -> str | None:
    """Inverse of :func:`phased_run_dirname`; ``None`` if not a phased-run dir."""
    return dirname[: -len(RUN_SUFFIX)] if dirname.endswith(RUN_SUFFIX) else None


def remote_generated_root(remote_run_dir: str) -> str:
    return f"{remote_run_dir.rstrip('/')}/{GENERATED_SUBDIR}"


def remote_artifact_paths(remote_run_dir: str, run_name: str) -> dict[str, str]:
    """Remote ``(metric, run_info, cost_record)`` paths under a pack run dir."""
    base = f"{remote_generated_root(remote_run_dir)}/{phased_run_dirname(run_name)}"
    return {
        "metric": f"{base}/{TARGET_METRIC_NAME}",
        "run_info": f"{base}/{RUN_INFO_REL}",
        "cost_record": f"{base}/{COST_RECORD_NAME}",
    }


def local_artifact_paths(staging_root: Path, run_name: str) -> dict[str, Path]:
    """Local staging paths, mirroring a LOCAL generated/ tree."""
    base = Path(staging_root) / phased_run_dirname(run_name)
    return {
        "metric": base / TARGET_METRIC_NAME,
        "run_info": base / RUN_INFO_REL,
        "cost_record": base / COST_RECORD_NAME,
    }


def parse_generated_listing(listing: str) -> list[str]:
    """Run names from an ``ls -1`` of the remote generated/ dir (phased dirs only)."""
    names = []
    for line in listing.splitlines():
        entry = line.strip().rstrip("/")
        if not entry:
            continue
        run_name = run_name_from_dirname(entry)
        if run_name is not None:
            names.append(run_name)
    return sorted(set(names))


def enumerate_runs(conn, remote_run_dir: str) -> list[str]:
    """List child run names under a pack's remote generated/ dir."""
    chunks: list[str] = []
    conn.run_with_streaming(f"ls -1 {remote_generated_root(remote_run_dir)} 2>/dev/null", chunks.append)
    return parse_generated_listing("".join(chunks))


@dataclass
class StageResult:
    run_name: str
    staged_dir: Path
    have_metric: bool
    have_run_info: bool
    have_cost_record: bool = False

    @property
    def complete(self) -> bool:
        """Complete = harvestable; cost_record is OPTIONAL (absent for crashed runs)."""
        return self.have_metric and self.have_run_info


def stage_run(conn, remote_run_dir: str, run_name: str, staging_root: Path) -> StageResult:
    """Pull the artifact JSONs for one run; tolerate a missing metric/cost (crash/orphan)."""
    remote = remote_artifact_paths(remote_run_dir, run_name)
    local = local_artifact_paths(staging_root, run_name)
    for path in local.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    for key in ("metric", "run_info", "cost_record"):
        if conn.exists(remote[key]):
            conn.get_file(remote[key], str(local[key]))
            have[key] = True
        else:
            have[key] = False
    return StageResult(
        run_name=run_name,
        staged_dir=local["metric"].parent,
        have_metric=have["metric"],
        have_run_info=have["run_info"],
        have_cost_record=have["cost_record"],
    )


@dataclass
class PackStage:
    run_id: str
    remote_run_dir: str
    results: list[StageResult] = field(default_factory=list)


def stage_pack(conn, remote_run_dir: str, staging_root: Path, run_id: str = "") -> PackStage:
    pack = PackStage(run_id=run_id, remote_run_dir=remote_run_dir)
    for run_name in enumerate_runs(conn, remote_run_dir):
        pack.results.append(stage_run(conn, remote_run_dir, run_name, staging_root))
    return pack


def resolve_remote_run_dir(registry, run_id: str) -> str:
    run = registry.find_run(run_id=run_id)
    if run is None:
        raise KeyError(f"run_id not found in registry: {run_id}")
    return run["remote_run_dir"]


def stage_packs(
    conn, registry, run_ids: Iterable[str], staging_root: Path,
    log: Callable[[str], None] = print,
) -> list[PackStage]:
    staged = []
    for run_id in run_ids:
        rrd = resolve_remote_run_dir(registry, run_id)
        pack = stage_pack(conn, rrd, staging_root, run_id=run_id)
        n_ok = sum(1 for r in pack.results if r.complete)
        log(f"[stage] {run_id}: {n_ok}/{len(pack.results)} runs complete ({rrd})")
        staged.append(pack)
    return staged


# --------------------------------------------------------------------------- CLI


def _connect_slurmech(profile: str):
    sys.path.insert(0, "/home/yigit/repos/research_stuff/shaq-workspace/slurmech")
    from slurmech.cli import _connect
    from slurmech.registry import Registry

    config, _creds, conn, _layout = _connect(profile)
    return conn, Registry(config.profile)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="mimarsinan-xlog1")
    parser.add_argument("--run-id", action="append", dest="run_ids", required=True,
                        help="slurmech pack run_id (repeatable)")
    parser.add_argument("--staging-root", default="runs/campaign/v2_staging/generated")
    args = parser.parse_args(argv)

    staging_root = Path(args.staging_root)
    conn, registry = _connect_slurmech(args.profile)
    packs = stage_packs(conn, registry, args.run_ids, staging_root)

    total = sum(len(p.results) for p in packs)
    complete = sum(1 for p in packs for r in p.results if r.complete)
    print(f"staged {complete}/{total} runs -> {staging_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
