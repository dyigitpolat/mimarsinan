"""Tests for :class:`SnapshotExecutor` — the single-worker queue that
decouples step-end snapshot bookkeeping from the pipeline thread.

Design invariants asserted here:

* Jobs submitted from the caller's thread are run on a dedicated worker
  thread (never on the submitter's thread).
* Jobs are executed serially, in submission order (single-worker queue).
* A job raising an exception does not kill the worker; subsequent jobs
  still run.
* ``wait_idle`` blocks until all queued work is drained.
* ``shutdown`` joins the worker cleanly; jobs submitted after shutdown
  are rejected (do not silently vanish).
"""

from __future__ import annotations

import threading
import time

import pytest

from mimarsinan.gui.snapshot_executor import SnapshotExecutor


class TestSnapshotExecutorBasics:
    def test_runs_job_on_worker_thread_not_caller(self) -> None:
        ex = SnapshotExecutor()
        try:
            caller_ident = threading.get_ident()
            seen: dict[str, int] = {}
            done = threading.Event()

            def job() -> None:
                seen["tid"] = threading.get_ident()
                done.set()

            ex.submit(job)
            assert done.wait(2.0), "job never ran"
            assert seen["tid"] != caller_ident
        finally:
            ex.shutdown()

    def test_submit_does_not_block_caller(self) -> None:
        ex = SnapshotExecutor()
        try:
            started = threading.Event()
            release = threading.Event()

            def slow_job() -> None:
                started.set()
                release.wait(2.0)

            t0 = time.monotonic()
            ex.submit(slow_job)
            elapsed = time.monotonic() - t0
            # Submit must return immediately even while a slow job runs.
            assert elapsed < 0.2, f"submit blocked for {elapsed:.3f}s"
            assert started.wait(2.0)
            release.set()
        finally:
            ex.shutdown()

    def test_jobs_execute_in_submission_order(self) -> None:
        ex = SnapshotExecutor()
        try:
            order: list[int] = []
            gate = threading.Event()

            def make_job(i: int):
                def job() -> None:
                    if i == 0:
                        gate.wait(2.0)
                    order.append(i)
                return job

            for i in range(5):
                ex.submit(make_job(i))
            gate.set()
            assert ex.wait_idle(timeout=2.0)
            assert order == [0, 1, 2, 3, 4]
        finally:
            ex.shutdown()


class TestSnapshotExecutorFaultTolerance:
    def test_failing_job_does_not_kill_worker(self) -> None:
        ex = SnapshotExecutor()
        try:
            ran: list[str] = []

            def boom() -> None:
                raise RuntimeError("kaboom")

            def ok() -> None:
                ran.append("ok")

            ex.submit(boom)
            ex.submit(ok)
            assert ex.wait_idle(timeout=2.0)
            assert ran == ["ok"]
        finally:
            ex.shutdown()


class TestSnapshotExecutorLifecycle:
    def test_wait_idle_returns_true_when_drained(self) -> None:
        ex = SnapshotExecutor()
        try:
            counter: list[int] = []
            for _ in range(3):
                ex.submit(lambda: counter.append(1))
            assert ex.wait_idle(timeout=2.0) is True
            assert len(counter) == 3
        finally:
            ex.shutdown()

    def test_shutdown_joins_worker(self) -> None:
        ex = SnapshotExecutor()
        ex.submit(lambda: None)
        ex.shutdown(timeout=2.0)
        assert not ex.is_alive()

    def test_submit_after_shutdown_raises(self) -> None:
        ex = SnapshotExecutor()
        ex.shutdown()
        with pytest.raises(RuntimeError):
            ex.submit(lambda: None)
