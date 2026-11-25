""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cl6_common as K

RA = HERE.parents[3]
E2E = RA / "12_tcad_readiness" / "e2e"
DRIVER = E2E / "driver"
ROWS = E2E / "routeb" / "rows"
PREFIX = "routeb:"

W34_START = "Readout Law"
TARGET_METRIC = "__target_metric"
RUN_CONFIG_DIR = "_RUN_CONFIG"


ROW_OF_RECORD_ABSENT = "W34_ROW_OF_RECORD_ABSENT"
SNAPSHOT_ROWS_ROOT = "W34_SNAPSHOT_ROWS_ROOT"

SEAL_ABSENT = "W34_SEAL_ABSENT"

ROW_FILES = ("config.json", "pipeline_stdout.log", "quantizer_state.json", "result.json")
PROFILE = re.compile(r"^\[PROFILE\] step='(?P<step>[^']+)' wall=\s*[0-9.]+s metric=(?P<metric>-?[0-9.]+) ")


class RouteBMemberError(ValueError):
    ""


class SnapshotCutError(ValueError):
    ""


def is_routeb(name: str) -> bool:
    return str(name).startswith(PREFIX)


def row_key(name: str) -> str:
    if not is_routeb(name):
        raise RouteBMemberError(f"{name!r} is not a route (b) member (no {PREFIX!r} prefix)")
    return str(name)[len(PREFIX):]


def point_name(name: str, arm: Optional[str] = None, adc: Optional[int] = None) -> str:
    ""
    base = row_key(name)
    return base if arm is None else f"{base}__{arm}__adc{adc}"


def row_dir(name: str, rows_root: Optional[Path] = None) -> Path:
    d = Path(rows_root or ROWS) / row_key(name)
    if not (d / "config.json").is_file():
        raise RouteBMemberError(f"route (b) row {row_key(name)!r} has no config.json under {d.parent}")
    return d


def rows_root_of(rows_root: Optional[Path] = None) -> str:
    ""
    return str(Path(rows_root or ROWS).resolve())


def row_of_record(name: str, rows_root: Optional[Path] = None) -> Path:
    ""
    path = row_dir(name, rows_root) / "result.json"
    status = json.loads(path.read_text()).get("status") if path.is_file() else None
    if status != "complete":
        raise RouteBMemberError(f"{ROW_OF_RECORD_ABSENT}: {path} is {'absent' if status is None else f'status {status!r}'}"
                                ": the row has no row of record (a cut or a count from it would be a silent wrong "
                                "source)")
    return path


def row_files(name: str, rows_root: Optional[Path] = None) -> Dict[str, str]:
    d = row_dir(name, rows_root)
    return {f: K.sha256_file(d / f) for f in ROW_FILES if (d / f).is_file()}


def row_config(name: str, rows_root: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads((row_dir(name, rows_root) / "config.json").read_text())


def row_cores(name: str, rows_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    ""
    pc = row_config(name, rows_root)["platform_constraints"]
    cores = (pc.get("target_overrides") or {}).get("cores", {}).get("value") or pc.get("cores")
    if not cores:
        raise RouteBMemberError(f"route (b) row {row_key(name)!r} declares no cores")
    return copy.deepcopy(cores)


def driver_modules():
    ""
    for path in (DRIVER / "members", DRIVER):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import e01_provider
    import e2e_members

    return e01_provider


def seal_paths() -> tuple:
    ""
    driver_modules()
    import common

    return tuple(common.SEALED_PATHS)


def require_seal() -> None:
    ""
    import inspect

    from mimarsinan.pipelining.session import PipelineSession

    if "sealed_paths" not in inspect.signature(PipelineSession.from_config).parameters:
        raise RouteBMemberError(f"{SEAL_ABSENT}: PipelineSession.from_config on this tree takes no sealed_paths, so a "
                                "route (b) session cannot open with the e2e seal the rows ran with (run the route (b) "
                                "export, head + e2e/code.patch)")


def open_session(cfg: Dict[str, Any]):
    ""
    require_seal()
    from mimarsinan.pipelining.session import PipelineSession

    return PipelineSession.from_config(cfg, sealed_paths=seal_paths())


def _held_out_batch(datasets_path: str) -> Optional[Path]:
    path = Path(datasets_path) / "cifar-100-python" / "test"
    return path if path.exists() else None


def member(name: str, *, start_step: str, provider: Optional[str] = None,
           rows_root: Optional[Path] = None) -> Dict[str, Any]:
    ""
    e01 = driver_modules()
    d = row_dir(name, rows_root)
    cfg = json.loads((d / "config.json").read_text())
    chosen = provider or cfg["data_provider_name"]
    if chosen not in e01.PROVIDERS:
        raise RouteBMemberError(f"{chosen!r} is not a route (b) provider ({sorted(e01.PROVIDERS)}): a route (b) "
                                "point reads its count on a route (b) provider's test slot, the E01 split")
    held = _held_out_batch(cfg["datasets_path"])
    if held is not None:
        raise RouteBMemberError(f"{held} exists: the row's datasets path holds a CIFAR-100 test batch, which no route "
                                "(b) point may sit beside (the E01 provider reads the training batch only)")

    def build() -> Dict[str, Any]:
        out = copy.deepcopy(cfg)
        out["data_provider_name"] = chosen
        out["experiment_name"] = point_name(name)
        return out

    K.MEMBERS[name] = (build, start_step)
    K.DATASETS = cfg["datasets_path"]
    return {"member": name, "row": d.name, "row_dir": str(d), "config_sha256": K.sha256_file(d / "config.json"),
            "provider": chosen, "row_provider": cfg["data_provider_name"], "datasets_path": cfg["datasets_path"],
            "start_step": start_step, "rows_root": rows_root_of(rows_root)}


def _sha256(path: Path) -> str:
    return K.sha256_file(path)


def _step_of(key: str, steps: Sequence[str]) -> Optional[str]:
    ""
    owners = [s for s in steps if key.startswith(s + ".")]
    return max(owners, key=len) if owners else None


def _payload_path(work: Path, strategy: str, filename: str) -> Path:
    from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache

    if strategy not in PipelineCache.LOAD_STORE_STRATEGIES:
        raise SnapshotCutError(f"cache entry file {filename!r} has an unknown load-store strategy {strategy!r}")
    return Path(PipelineCache.LOAD_STORE_STRATEGIES[strategy](filename).path(str(work)))


def _run_scoped() -> tuple:
    from mimarsinan.tuning.orchestration import run_ledger

    return tuple(run_ledger.RUN_SCOPED_KEYS), (run_ledger.RETENTION_ENVELOPE_CACHE_KEY,)


def _ledger_files() -> tuple:
    from mimarsinan.chip_simulation.cost_extraction import FT_PASS_WALLS_FILENAME
    from mimarsinan.tuning.orchestration.run_instrumentation import RETENTION_LEDGER_FILENAME

    return RETENTION_LEDGER_FILENAME, FT_PASS_WALLS_FILENAME


def profile_metrics(log: Path) -> List[tuple]:
    ""
    out = []
    for line in Path(log).read_text(errors="replace").splitlines():
        m = PROFILE.match(line)
        if m:
            out.append((m["step"], float(m["metric"])))
    return out


def _target_metric_at_cut(log: Optional[Path], steps_run: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not steps_run:
        return None
    last = steps_run[-1]
    if log is None or not Path(log).is_file():
        raise SnapshotCutError(f"no session log ({log}) to read the metric {last!r} printed, which the pipeline's "
                               f"target metric holds at the cut")
    seen = profile_metrics(log)
    order = [s for s, _ in seen]
    if order[: len(steps_run)] != list(steps_run):
        raise SnapshotCutError(f"the session log's steps {order[:len(steps_run)]} are not the steps run before the cut "
                               f"{list(steps_run)} (step {last!r} missing or out of order): not this run's log")
    value = seen[len(steps_run) - 1][1]
    return {"value": value, "step": last, "source": str(log), "source_sha256": _sha256(Path(log)),
            "precision": "the step's [PROFILE] metric, printed at 4 decimals (emit_step_profile)"}


def _cut_ledger(path: Path, later: set, kind: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if kind == "retention":
        entries = list(data.get("entries") or [])
        kept = [e for e in entries if e.get("step") not in later]
        dropped = [e.get("step") for e in entries if e.get("step") in later]
        payload = {"entries": kept} if kept else None
    else:
        passes = list(data.get("passes") or [])
        kept = [p for p in passes if str(p.get("label", "")).split("/", 1)[0] not in later]
        dropped = [p.get("label") for p in passes if str(p.get("label", "")).split("/", 1)[0] in later]
        payload = {"max_ft_pass_wall_s": max(float(p["wall_s"]) for p in kept), "passes": kept} if kept else None
    if payload is None:
        path.unlink()
    elif dropped:
        path.write_text(json.dumps(payload, indent=2))
    return {"dropped_steps" if kind == "retention" else "dropped_passes": dropped, "kept": len(kept),
            "file": "removed (empty at the cut)" if payload is None else ("cut" if dropped else "unchanged")}


def cut_run(work: Path, snap: Path, steps: Sequence[str], start: str, log: Optional[Path],
            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ""
    work, snap = Path(work), Path(snap)
    steps = list(steps)
    if start not in steps:
        raise SnapshotCutError(f"start step {start!r} is not a step of the pipeline {steps}")
    run = snap / "run"
    if run.exists():
        raise SnapshotCutError(f"{run} exists: a snapshot is written once")
    meta_src = json.loads((work / "metadata.json").read_text())
    steps_run, later = steps[: steps.index(start)], set(steps[steps.index(start):])
    run_scoped, write_once = _run_scoped()
    order = list(meta_src)
    kept_at = [i for i, k in enumerate(order) if _step_of(k, steps) in steps_run]
    dropped_at = [i for i, k in enumerate(order) if _step_of(k, steps) in later]
    last_kept, first_dropped = max(kept_at, default=-1), min(dropped_at, default=len(order))
    if last_kept > first_dropped:
        raise SnapshotCutError(f"the cache's order interleaves kept and dropped steps ({order[first_dropped]!r} before "
                               f"{order[last_kept]!r}): not one run's metadata")
    target = _target_metric_at_cut(log, steps_run)
    keep, drop, placed = {}, {}, {}
    for i, key in enumerate(order):
        owner = _step_of(key, steps)
        if owner is not None:
            (keep if owner in steps_run else drop)[key] = meta_src[key]
            continue
        if key == TARGET_METRIC:
            (keep if target is not None else drop)[key] = meta_src[key]
            placed[key] = "restored to the last kept step's printed metric" if target else "dropped: no step ran"
            continue
        if i < last_kept:
            if key in run_scoped and key not in write_once:
                raise SnapshotCutError(f"run-scoped ledger {key!r} was written before the cut and may have moved "
                                       "after it: its value at the cut is not in the finished run")
            keep[key] = meta_src[key]
            placed[key] = "kept: before an entry of a kept step"
        elif i > first_dropped:
            drop[key] = meta_src[key]
            placed[key] = "dropped: after an entry of a dropped step"
        else:
            raise SnapshotCutError(f"cache entry {key!r} lies between the last kept step's entries and the first "
                                   "dropped step's: the cache's order cannot place it before or after the cut")
    payload_files = {}
    for key, (strategy, filename) in meta_src.items():
        payload_files[key] = _payload_path(work, strategy, filename).name
    missing = sorted(k for k in keep if not (work / payload_files[k]).is_file())
    if missing:
        raise SnapshotCutError(f"kept cache entries without their payload file in {work}: {missing}")
    shutil.copytree(work, run)
    try:
        for key in drop:
            (run / payload_files[key]).unlink(missing_ok=True)
        if target is not None:
            strategy, filename = meta_src[TARGET_METRIC]
            _payload_path(run, strategy, filename).write_text(json.dumps(target["value"]))
        retention, walls = _ledger_files()
        ledgers = {retention: _cut_ledger(run / retention, later, "retention"),
                   walls: _cut_ledger(run / walls, later, "walls")}
        kept_files = {payload_files[k] for k in keep} | {"metadata.json", retention, walls}
        files_dropped = {}
        for p in sorted(run.iterdir()):
            if p.name == RUN_CONFIG_DIR or p.name in kept_files:
                continue
            files_dropped[p.name] = _sha256(p) if p.is_file() else "directory"
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        (run / "metadata.json").write_text(json.dumps(keep))
    except BaseException:
        shutil.rmtree(run, ignore_errors=True)
        raise
    manifest = {
        "format": "cl6-routeb-snapshot/1", "workdir": str(work), "start_step": start, "steps": steps,
        "steps_run": steps_run, "cache_entries_kept": keep, "cache_entries_dropped": drop,
        "unprefixed_entries": placed, "target_metric_at_cut": target,
        "ledgers": {k: v for k, v in ledgers.items() if v is not None}, "files_dropped": files_dropped,
        "files_sha256": {p.name: _sha256(p) for p in sorted(run.iterdir()) if p.is_file()},
        **(extra or {}),
    }
    K.write_json(snap / "snapshot.json", manifest)
    return manifest


def pipeline_steps(name: str, scratch: Path) -> List[str]:
    ""
    cfg = K.point_config(name, "untrained", K.AT_LEAST_A, row_cores(name), allow_scheduling=False,
                         name=f"{point_name(name)}__steps")
    cfg["_working_directory"] = str(scratch)
    session = open_session(cfg)
    try:
        return [step for step, _ in session.pipeline.steps]
    finally:
        session.finish()


def cut(name: str, work: Path, snap: Path, *, start: str = W34_START, provider: Optional[str] = None,
        log: Optional[Path] = None, rows_root: Optional[Path] = None) -> Dict[str, Any]:
    ""
    info = member(name, start_step=start, provider=provider, rows_root=rows_root)
    row_of_record(name, rows_root)
    if (Path(snap) / "run").exists():
        raise SnapshotCutError(f"{Path(snap) / 'run'} exists: a snapshot is written once")
    scratch = Path(snap) / "steps_probe"
    try:
        steps = pipeline_steps(name, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    log = Path(log) if log else Path(info["row_dir"]) / "pipeline_stdout.log"
    return cut_run(work, snap, steps, start, log, extra={
        "member": name, "routeb": info, "provider": info["provider"], "src_digest": K.src_digest(), "wt": str(K.WT),
        "rows_root": info["rows_root"], "row_files_sha256": row_files(name, rows_root),
        "label": "a cut of a route (b) working directory (cl6_routeb.cut)"})


def e01_count(record: Dict[str, Any], provider_name: str, provider: Any = None,
              datasets_path: Optional[str] = None) -> Dict[str, Any]:
    ""
    e01 = driver_modules()
    if provider_name not in e01.PROVIDERS:
        raise RouteBMemberError(f"{provider_name!r} is not a route (b) provider ({sorted(e01.PROVIDERS)}): its test "
                                "slot is no E01 split")
    if provider is None:
        provider = e01.PROVIDERS[provider_name](str(datasets_path))
    n = len(provider.indices["test"])
    accuracy = record["accuracy"]
    metric = float(accuracy["deployed"]["metric"])
    count = round(metric * n)
    if abs(metric * n - count) > 1e-6:
        raise RouteBMemberError(f"deployed metric {metric!r} on {n} images is no whole count ({metric * n!r})")
    measured = [r for r in accuracy.get("reads") or [] if r.get("kind") == "measured"]
    return {"provider": provider_name, "images": n, "correct": int(count), "metric": metric,
            "split": "the provider's test slot: the E01 images (drawn from the CIFAR-100 training batch), never a "
                     "dataset test split",
            "test_indices_sha256": provider.digests().get("test_indices_sha256"),
            "deployed": dict(accuracy["deployed"]), "measured_read": measured[-1] if measured else None}


def row_counts(name: str, rows_root: Optional[Path] = None, *, require: bool = False) -> Optional[Dict[str, Any]]:
    ""
    path = row_of_record(name, rows_root) if require else row_dir(name, rows_root) / "result.json"
    if not path.is_file():
        return None
    res = json.loads(path.read_text())
    return {"result_sha256": K.sha256_file(path), "K9_correct": (res.get("K9") or {}).get("correct"),
            "K9_images": (res.get("K9") or {}).get("images"), "K14": res.get("K14")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cut")
    c.add_argument("row")
    c.add_argument("workdir", type=Path)
    c.add_argument("snapshot", type=Path)
    c.add_argument("--start", default=W34_START)
    c.add_argument("--provider", default=None)
    c.add_argument("--log", type=Path, default=None)
    c.add_argument("--rows-root", type=Path, default=None)
    args = ap.parse_args(argv)
    name = args.row if is_routeb(args.row) else PREFIX + args.row
    manifest = cut(name, args.workdir, args.snapshot, start=args.start, provider=args.provider, log=args.log,
                   rows_root=args.rows_root)
    print(f"snapshot {args.snapshot}: kept {len(manifest['cache_entries_kept'])} entries, dropped "
          f"{len(manifest['cache_entries_dropped'])}; files dropped {sorted(manifest['files_dropped'])}; target metric "
          f"at the cut {manifest['target_metric_at_cut']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
