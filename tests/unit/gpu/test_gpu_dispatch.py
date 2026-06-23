"""Locks the GPU dispatcher: every job runs once, leases honor capacity, no real GPU."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gpu"))

import gpu_lease as gl  # noqa: E402
import gpu_dispatch as gd  # noqa: E402


def _two_free_gpus():
    return [gl.GpuStat(0, 100000, 100000, 0), gl.GpuStat(1, 100000, 100000, 0)]


def test_dispatch_runs_every_job_and_records_results(tmp_path):
    jobs = [{"id": f"j{i}", "mode": "fit", "need_mb": 8000,
             "cmd": [sys.executable, "-c", "import os;print(os.environ['CUDA_VISIBLE_DEVICES'])"]}
            for i in range(5)]
    results = gd.dispatch(jobs, results_path=str(tmp_path / "res.json"),
                          logdir=str(tmp_path / "logs"),
                          snapshot=_two_free_gpus, directory=str(tmp_path / "leases"),
                          poll=0.01)
    assert {r["id"] for r in results} == {f"j{i}" for i in range(5)}
    assert all(r["returncode"] == 0 for r in results)
    # each job saw a concrete device id (0 or 1) -> CUDA_VISIBLE_DEVICES was set
    for i in range(5):
        log = (tmp_path / "logs" / f"j{i}.log").read_text().strip()
        assert log in {"0", "1"}


def test_dispatch_drains_when_more_jobs_than_gpus(tmp_path):
    # 1 fit-GPU of capacity, 3 jobs each needing the whole card -> must serialize,
    # but all three still complete (a freed GPU is grabbed by the next pending job).
    snap = lambda: [gl.GpuStat(0, 100000, 100000, 0)]
    jobs = [{"id": f"k{i}", "mode": "fit", "need_mb": 90000,
             "cmd": [sys.executable, "-c", "pass"]} for i in range(3)]
    results = gd.dispatch(jobs, snapshot=snap, directory=str(tmp_path / "l"),
                          logdir=str(tmp_path / "lg"), poll=0.01)
    assert len(results) == 3 and all(r["returncode"] == 0 for r in results)
    # leases all released at the end
    assert not gl.read_leases(str(tmp_path / "l"))


def test_dispatch_reports_nonzero_returncode(tmp_path):
    jobs = [{"id": "boom", "mode": "fit", "need_mb": 1,
             "cmd": [sys.executable, "-c", "import sys;sys.exit(3)"]}]
    results = gd.dispatch(jobs, snapshot=_two_free_gpus,
                          directory=str(tmp_path / "l"), logdir=str(tmp_path / "lg"),
                          poll=0.01)
    assert results[0]["returncode"] == 3
