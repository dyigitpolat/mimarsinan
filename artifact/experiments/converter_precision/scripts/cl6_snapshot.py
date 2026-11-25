""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cl6_common as K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("member", choices=sorted(K.MEMBERS))
    ap.add_argument("out")
    ap.add_argument("--arm", default="tuned", choices=sorted(K.ARMS))
    ap.add_argument("--adc", type=int, default=K.AT_LEAST_A)
    ap.add_argument("--cores", default=None, help="JSON list of core types")
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()

    gpu_fraction = K.apply_gpu_cap()
    from mimarsinan.pipelining.session import PipelineSession

    out = Path(args.out)
    run_dir = out / "run"
    if run_dir.exists():
        raise SystemExit(f"refused: {run_dir} exists (a snapshot is written once)")
    declared = K.base_config(args.member)["platform_constraints"]["cores"]
    cores = json.loads(args.cores) if args.cores else declared
    if args.scale is not None:
        cores = [{**ct, "count": max(1, int(ct["count"] * args.scale))} for ct in cores]
    cfg = K.point_config(args.member, args.arm, args.adc, cores, name=f"{args.member}__snapshot")
    cfg["_working_directory"] = str(run_dir)
    start = K.start_step(args.member)
    t0 = time.time()
    session = PipelineSession.from_config(cfg)
    names = [name for name, _ in session.pipeline.steps]
    if start not in names:
        raise SystemExit(f"refused: start step {start!r} is not in the member's pipeline {names}")
    stop = names[names.index(start) - 1]
    print(f"[cl6_snapshot] {args.member}: steps {names}; stop after {stop!r} (start step {start!r})", flush=True)
    try:
        session.surface_config_advisories()
        session.pipeline.run(stop_step=stop)
    finally:
        session.finish()
    wall = time.time() - t0
    files = {p.name: K.sha256_file(p) for p in sorted(run_dir.iterdir()) if p.is_file()}
    metadata = json.loads((run_dir / "metadata.json").read_text())
    K.write_json(out / "snapshot.json", {
        "member": args.member, "arm": args.arm, "adc_bits": args.adc, "cores": cores,
        "start_step": start, "stopped_after": stop, "steps": names,
        "steps_run": names[: names.index(start)], "cache_entries": metadata, "files_sha256": files,
        "target_metric": json.loads((run_dir / "__target_metric.json").read_text()),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "src_digest": K.src_digest(), "wt": str(K.WT), "wall_s": wall,
        "gpu": K.gpu_peak(gpu_fraction),
    })
    print(f"[cl6_snapshot] done in {wall:.1f} s: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
