"""The autonomous, never-idle GPU runner for the publication campaign.

Drains the persistent :class:`GpuQueue` FOREVER (until a STOP sentinel), keeping
every leasable GPU full. Unlike the v1 dispatcher it does NOT exit when the queue
empties — it idles-poll, so the instant a producer enqueues more work the GPUs fill
again. It writes a ``status.json`` heartbeat every loop so progress is observable
WITHOUT a human polling, and the `campaign_watch` companion wakes the operator at a
low-watermark *before* the GPUs starve.

Robust finalized-run detection (the v1 weakness): a job is finalized when its
subprocess exits; the recorded result carries returncode, wall, AND whether the
expected output artifact appeared (``expect_artifact``), so a process that exits 0
but wrote nothing is flagged rather than silently counted done. Per-job timeouts
kill hung runs; orphaned leases (dead pids) and orphaned running/ jobs (crashed
runner) are reclaimed on startup.

Run:  python scripts/gpu/campaign_runner.py [--poll 3] [--max-per-gpu 2]
Stop: create q/STOP  (or python scripts/gpu/campaign_runner.py --stop)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(__file__))
import gpu_lease as gl
from gpu_queue import GpuQueue, campaign_dir


def _now() -> float:
    return time.time()


class Runner:
    def __init__(self, queue: GpuQueue, *, poll: float = 3.0, max_per_gpu: int = 2,
                 snapshot: Callable[[], List[gl.GpuStat]] = gl.query_nvidia_smi,
                 lease_dir=None, status_path=None, logdir=None):
        self.q = queue
        self.poll = poll
        self.max_per_gpu = max_per_gpu
        self.snapshot = snapshot
        self.lease_dir = lease_dir or gl.lease_dir()
        self.status_path = status_path or os.path.join(campaign_dir(), "status.json")
        self.logdir = logdir or os.path.join(campaign_dir(), "logs")
        os.makedirs(self.logdir, exist_ok=True)
        self.running: Dict[str, dict] = {}  # running_path -> {proc, lease, job, started, log}

    # -- reap finished / timed-out jobs --
    def _reap(self):
        for path in list(self.running):
            r = self.running[path]
            proc, job = r["proc"], r["job"]
            timeout = float(job.get("timeout_s") or 0)
            elapsed = _now() - r["started"]
            timed_out = timeout and elapsed > timeout
            if proc.poll() is None and not timed_out:
                continue
            if proc.poll() is None and timed_out:
                proc.kill()
                try:
                    proc.wait(timeout=30)
                except Exception:
                    pass
            gl.release(r["lease"])
            try:
                r["log"].close()
            except Exception:
                pass
            rc = proc.returncode
            artifact = job.get("expect_artifact")
            artifact_ok = (artifact is None) or os.path.exists(
                os.path.join(job.get("cwd", "."), artifact))
            success = (rc == 0) and artifact_ok and not timed_out
            result = {"returncode": rc, "wall_s": round(elapsed, 1), "gpu": r["lease"].gpu,
                      "timed_out": bool(timed_out), "artifact_ok": artifact_ok,
                      "finished_at": _now()}
            self.q.finish(path, result, success)
            del self.running[path]

    # -- fill free GPUs from the queue (lease-then-claim so a job is never dequeued
    # without a GPU to run it on) --
    def _fill(self):
        while not self.q.stop_requested():
            names = self.q.pending_names()
            if not names:
                return
            placed = False
            for name in names:
                ppath = os.path.join(self.q._dir("pending"), name)
                try:
                    with open(ppath) as fh:
                        job = json.load(fh)
                except (OSError, ValueError):
                    continue
                lease = gl.acquire(job.get("mode", "fit"), int(job.get("need_mb", 8000)),
                                   directory=self.lease_dir, snapshot=self.snapshot,
                                   cmd=str(job.get("id", "")), max_per_gpu=self.max_per_gpu)
                if lease is None:
                    continue  # this job's mode can't be placed now; try the next one
                dst = os.path.join(self.q._dir("running"), name)
                try:
                    os.rename(ppath, dst)        # atomic claim
                except OSError:
                    gl.release(lease)
                    continue
                self._launch(job, dst, lease)
                placed = True
                break
            if not placed:
                return  # no GPU could be leased for any pending job

    def _launch(self, job, running_path, lease):
        cmd = job["cmd"]
        cmd = cmd if isinstance(cmd, list) else ["bash", "-lc", cmd]
        env = dict(os.environ, **(job.get("env") or {}))
        env["CUDA_VISIBLE_DEVICES"] = str(lease.gpu)
        log = open(os.path.join(self.logdir, f"{job['id']}.log"), "w")
        proc = subprocess.Popen(cmd, cwd=job.get("cwd"), env=env, stdout=log,
                                stderr=subprocess.STDOUT, text=True)
        self.running[running_path] = {"proc": proc, "lease": lease, "job": job,
                                      "started": _now(), "log": log}

    def _heartbeat(self):
        c = self.q.counts()
        try:
            gpus = [{"i": s.index, "free_mb": s.mem_free, "util": s.util}
                    for s in self.snapshot()]
        except Exception:
            gpus = []
        status = {
            "ts": _now(),
            "pending": c["pending"], "running": c["running"],
            "done": c["done"], "failed": c["failed"],
            "drained": c["pending"] == 0 and c["running"] == 0,
            "running_jobs": [{"id": r["job"]["id"], "gpu": r["lease"].gpu,
                              "elapsed_s": round(_now() - r["started"], 1)}
                             for r in self.running.values()],
            "gpus": gpus,
        }
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(status, fh, indent=2)
        os.replace(tmp, self.status_path)

    def run(self):
        self.q.requeue_running()  # reclaim a crashed runner's in-flight jobs
        while True:
            self._reap()
            if self.q.stop_requested():
                if not self.running:
                    self._heartbeat()
                    return
            elif not self.q.pause_requested():
                self._fill()
            self._heartbeat()
            time.sleep(self.poll)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poll", type=float, default=3.0)
    p.add_argument("--max-per-gpu", type=int, default=2)
    p.add_argument("--stop", action="store_true", help="signal the running runner to stop")
    args = p.parse_args(argv)
    q = GpuQueue()
    if args.stop:
        q.request_stop()
        print("STOP requested; the runner will drain in-flight jobs then exit.")
        return 0
    runner = Runner(q, poll=args.poll, max_per_gpu=args.max_per_gpu)
    signal.signal(signal.SIGTERM, lambda *_: q.request_stop())
    print(f"campaign_runner: draining {q.root} (poll {args.poll}s, max/gpu {args.max_per_gpu})")
    runner.run()
    print("campaign_runner: stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
