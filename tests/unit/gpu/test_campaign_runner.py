"""Locks the autonomous campaign queue + runner + watcher (no real GPU; snapshots injected)."""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))

import gpu_lease as gl  # noqa: E402
from gpu_queue import GpuQueue  # noqa: E402
import campaign_runner as cr  # noqa: E402
import campaign_watch as cw  # noqa: E402


def two_gpus():
    return [gl.GpuStat(0, 100000, 100000, 0), gl.GpuStat(1, 100000, 100000, 0)]


# --------------------------------------------------------------------------- #
# queue: ordering, atomic claim, lifecycle, crash recovery
# --------------------------------------------------------------------------- #

def test_queue_orders_by_priority_then_time(tmp_path):
    q = GpuQueue(str(tmp_path / "q"))
    q.enqueue({"id": "late_cheap", "priority": 10, "cmd": ["true"]})
    q.enqueue({"id": "expensive", "priority": 90, "cmd": ["true"]})
    job, _ = q.claim_next()
    assert job["id"] == "late_cheap"  # lowest priority number runs first


def test_queue_claim_is_atomic_and_lifecycle(tmp_path):
    q = GpuQueue(str(tmp_path / "q"))
    q.enqueue({"id": "j", "cmd": ["true"]})
    job, path = q.claim_next()
    assert q.counts()["running"] == 1 and q.counts()["pending"] == 0
    assert q.claim_next() is None  # already claimed
    q.finish(path, {"returncode": 0}, success=True)
    assert q.counts()["done"] == 1 and q.counts()["running"] == 0


def test_requeue_running_recovers_a_crashed_runner(tmp_path):
    q = GpuQueue(str(tmp_path / "q"))
    q.enqueue({"id": "j", "cmd": ["true"]})
    q.claim_next()
    assert q.requeue_running() == 1
    assert q.counts()["pending"] == 1 and q.counts()["running"] == 0


# --------------------------------------------------------------------------- #
# runner: actually drains the queue across GPUs and NEVER exits until STOP
# --------------------------------------------------------------------------- #

def test_runner_drains_queue_then_idles_until_stop(tmp_path):
    q = GpuQueue(str(tmp_path / "q"))
    for i in range(5):
        q.enqueue({"id": f"j{i}", "mode": "fit", "need_mb": 8000,
                   "cmd": [sys.executable, "-c", "pass"]})
    runner = cr.Runner(q, poll=0.02, max_per_gpu=2, snapshot=two_gpus,
                       lease_dir=str(tmp_path / "leases"),
                       status_path=str(tmp_path / "status.json"),
                       logdir=str(tmp_path / "logs"))
    th = threading.Thread(target=runner.run, daemon=True)
    th.start()
    # wait until all 5 are done (runner keeps running — proves it does not exit on drain)
    for _ in range(500):
        if q.counts()["done"] == 5:
            break
        time.sleep(0.02)
    assert q.counts()["done"] == 5
    assert th.is_alive()  # still draining-poll, not exited
    # a late enqueue is still picked up (the never-idle property)
    q.enqueue({"id": "late", "mode": "fit", "cmd": [sys.executable, "-c", "pass"]})
    for _ in range(500):
        if q.counts()["done"] == 6:
            break
        time.sleep(0.02)
    assert q.counts()["done"] == 6
    q.request_stop()
    th.join(timeout=5)
    assert not th.is_alive()
    # leases all released; status heartbeat written
    assert not gl.read_leases(str(tmp_path / "leases"))
    assert json.load(open(str(tmp_path / "status.json")))["done"] == 6


def test_runner_pause_stops_launching_but_keeps_reaping(tmp_path):
    q = GpuQueue(str(tmp_path / "q"))
    for i in range(3):
        q.enqueue({"id": f"j{i}", "mode": "fit", "cmd": [sys.executable, "-c", "pass"]})
    open(q.pause_path, "w").close()  # PAUSE before the runner starts
    runner = cr.Runner(q, poll=0.02, snapshot=two_gpus, lease_dir=str(tmp_path / "l"),
                       status_path=str(tmp_path / "s.json"), logdir=str(tmp_path / "lg"))
    th = threading.Thread(target=runner.run, daemon=True); th.start()
    for _ in range(50):
        time.sleep(0.02)
    assert q.counts()["pending"] == 3 and q.counts()["running"] == 0  # nothing launched while paused
    os.remove(q.pause_path)  # unpause -> drains
    for _ in range(500):
        if q.counts()["done"] == 3:
            break
        time.sleep(0.02)
    assert q.counts()["done"] == 3
    q.request_stop(); th.join(timeout=5)


def test_runner_flags_a_job_that_exits_clean_but_writes_no_artifact(tmp_path):
    q = GpuQueue(str(tmp_path / "q"))
    q.enqueue({"id": "noout", "mode": "fit", "cmd": [sys.executable, "-c", "pass"],
               "cwd": str(tmp_path), "expect_artifact": "missing.json"})
    runner = cr.Runner(q, poll=0.02, snapshot=two_gpus, lease_dir=str(tmp_path / "l"),
                       status_path=str(tmp_path / "s.json"), logdir=str(tmp_path / "lg"))
    th = threading.Thread(target=runner.run, daemon=True); th.start()
    for _ in range(500):
        if q.counts()["failed"] == 1:
            break
        time.sleep(0.02)
    q.request_stop(); th.join(timeout=5)
    failed = q.list_state("failed")
    assert failed and failed[0]["result"]["artifact_ok"] is False


# --------------------------------------------------------------------------- #
# watcher: wakes at the low watermark and on drain
# --------------------------------------------------------------------------- #

def test_watch_wakes_on_low_watermark(tmp_path):
    sp = tmp_path / "status.json"
    sp.write_text(json.dumps({"pending": 3, "running": 4, "done": 1, "failed": 0}))
    ev = cw.watch(str(sp), low=8, stall=300, fail_burst=5, poll=0,
                  sleep=lambda _: None, clock=lambda: 0.0)
    assert ev["trigger"] == "refill" and ev["pending"] == 3


def test_watch_wakes_on_drain(tmp_path):
    sp = tmp_path / "status.json"
    sp.write_text(json.dumps({"pending": 0, "running": 0, "done": 9, "failed": 0}))
    ev = cw.watch(str(sp), low=8, stall=300, fail_burst=5, poll=0, sleep=lambda _: None)
    assert ev["trigger"] == "drained"
