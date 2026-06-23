"""Drain a manifest of GPU jobs across the free GPUs, keeping them saturated.

The shared queueing infra (minimal): given a list of jobs each tagged ``free``
(profiling — exclusive free GPU) or ``fit`` (correctness — any GPU that fits),
place as many as the GPUs allow, and the moment a GPU frees, grab it for the next
pending job — so a free GPU is never idle while work is queued. Concurrency is
bounded by real GPU capacity (via :mod:`gpu_lease`), not a fixed worker count, and
GPU ids are never hard-coded (the free set is discovered each round).

Manifest JSON: ``[{"id","mode","need_mb","cmd":[...]|str,"cwd"?,"env"?}, ...]``.
Per-job stdout+stderr go to ``<logdir>/<id>.log``; a results JSON is written
incrementally so progress survives interruption.

CLI:  python gpu_dispatch.py --manifest jobs.json --results out.json [--logdir d]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))
import gpu_lease as gl  # noqa: E402


def _as_cmd(cmd):
    return cmd if isinstance(cmd, list) else ["bash", "-lc", cmd]


def dispatch(jobs: List[dict], *, results_path: Optional[str] = None,
             logdir: Optional[str] = None,
             snapshot: Callable[[], List[gl.GpuStat]] = gl.query_nvidia_smi,
             directory: Optional[str] = None,
             poll: float = 4.0,
             max_per_gpu: Optional[int] = 2,
             sleep: Callable[[float], None] = time.sleep,
             clock: Callable[[], float] = time.monotonic,
             popen=subprocess.Popen) -> List[dict]:
    """Run every job exactly once across leased GPUs; return per-job result dicts."""
    directory = directory or gl.lease_dir()
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    pending = list(jobs)
    running: Dict[str, dict] = {}
    results: List[dict] = []

    def flush():
        if results_path:
            with open(results_path, "w") as fh:
                json.dump(results, fh, indent=2)

    while pending or running:
        placed_any = False
        still_pending = []
        for job in pending:
            mode = job.get("mode", "fit")
            need = int(job.get("need_mb", gl.DEFAULT_FIT_MB))
            lease = gl.acquire(mode, need, directory=directory, snapshot=snapshot,
                               cmd=str(job.get("id", "")), max_per_gpu=max_per_gpu)
            if lease is None:
                still_pending.append(job)
                continue
            env = dict(os.environ, **(job.get("env") or {}))
            env["CUDA_VISIBLE_DEVICES"] = str(lease.gpu)
            log = open(os.path.join(logdir, f"{job['id']}.log"), "w") if logdir \
                else subprocess.DEVNULL
            proc = popen(_as_cmd(job["cmd"]), cwd=job.get("cwd"), env=env,
                         stdout=log, stderr=subprocess.STDOUT, text=True)
            running[job["id"]] = {"proc": proc, "lease": lease, "log": log,
                                  "start": clock(), "gpu": lease.gpu}
            placed_any = True
        pending = still_pending

        done = [jid for jid, r in running.items() if r["proc"].poll() is not None]
        for jid in done:
            r = running.pop(jid)
            gl.release(r["lease"])
            if r["log"] not in (subprocess.DEVNULL, None):
                try:
                    r["log"].close()
                except Exception:
                    pass
            results.append({"id": jid, "gpu": r["gpu"],
                            "returncode": r["proc"].returncode,
                            "wall_s": round(clock() - r["start"], 2)})
            flush()

        if not placed_any and not done and (pending or running):
            sleep(poll)

    flush()
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", default=None)
    p.add_argument("--logdir", default=None)
    p.add_argument("--poll", type=float, default=4.0)
    p.add_argument("--max-per-gpu", type=int, default=2,
                   help="max concurrent fit jobs per GPU (avoid oversubscription)")
    args = p.parse_args(argv)
    with open(args.manifest) as fh:
        jobs = json.load(fh)
    results = dispatch(jobs, results_path=args.results, logdir=args.logdir,
                       poll=args.poll, max_per_gpu=args.max_per_gpu)
    ok = sum(1 for r in results if r["returncode"] == 0)
    print(f"dispatched {len(results)} jobs: {ok} ok, {len(results) - ok} failed")
    for r in results:
        if r["returncode"] != 0:
            print(f"  FAILED {r['id']} (gpu {r['gpu']}, rc {r['returncode']})")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
