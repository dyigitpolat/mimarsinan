"""Single-worker executor that decouples snapshot bookkeeping from the pipeline thread."""

from __future__ import annotations

import logging
import queue
import threading
import time as _time
from typing import Callable

logger = logging.getLogger("mimarsinan.gui.runtime.snapshot_executor")

Job = Callable[[], None]


class SnapshotExecutor:
    """FIFO, single-worker background executor for snapshot bookkeeping.

    The single worker guarantees snapshots persist and broadcast in submission order.
    """

    def __init__(self, name: str = "SnapshotExecutor") -> None:
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, job: Job) -> None:
        """Enqueue *job* for execution on the worker thread; raise if already shut down."""
        if self._stopped.is_set():
            raise RuntimeError("SnapshotExecutor has been shut down; cannot submit new jobs")
        self._queue.put(job)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until every submitted job has completed; return ``True`` if drained, ``False`` on timeout."""
        deadline: float | None = None
        if timeout is not None:
            deadline = _time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                if deadline is None:
                    self._queue.all_tasks_done.wait()
                else:
                    remaining = deadline - _time.monotonic()
                    if remaining <= 0:
                        return False
                    self._queue.all_tasks_done.wait(remaining)
        return True

    def shutdown(self, timeout: float | None = 5.0) -> None:
        """Signal the worker to stop after draining the queue and join it."""
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._queue.put(None)
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                try:
                    job()
                except Exception:
                    logger.exception("SnapshotExecutor job raised")
            finally:
                self._queue.task_done()
