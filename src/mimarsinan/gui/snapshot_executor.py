"""Single-worker executor that decouples snapshot bookkeeping from the
pipeline thread.

Background
----------
When a pipeline step completes, ``GUIHandle.on_step_end`` needs to:

1. Walk the pipeline state and build a JSON summary + lazy resource
   descriptors (``build_step_snapshot``).
2. Record the step on the ``DataCollector`` (triggers WebSocket
   broadcasts).
3. Persist the step to disk (``save_step_to_persisted``) — this writes
   potentially large ``steps.json`` files to the run directory.

Step 3 (and, for very large models, step 1) can take tens to hundreds of
milliseconds. Running them on the pipeline thread stalls the next step
and creates user-visible jank in live training/validation plots.

This module provides :class:`SnapshotExecutor`: a FIFO, single-worker
queue owned by the GUI layer. Hooks submit callables and return
immediately; the worker drains them in submission order. A failing job
does not kill the worker — we log and continue — because a single bad
step should not silently disable snapshot persistence for the rest of
the run.

The executor is deliberately single-worker: later steps' snapshots must
never land in ``steps.json`` before earlier ones (disk persistence uses
merge-by-name semantics, but WS ``step_completed`` events must also
arrive in order for the frontend progress view to make sense).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger("mimarsinan.gui.snapshot_executor")

Job = Callable[[], None]


class SnapshotExecutor:
    """FIFO, single-worker background executor for snapshot bookkeeping.

    Thread model
    ------------
    * One dedicated daemon worker thread drains a ``queue.Queue``.
    * ``submit`` is non-blocking and thread-safe.
    * ``wait_idle`` blocks until the queue is fully drained.
    * ``shutdown`` signals the worker to exit and joins it.
    """

    def __init__(self, name: str = "SnapshotExecutor") -> None:
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- Public API ----------------------------------------------------------

    def submit(self, job: Job) -> None:
        """Enqueue *job* for later execution on the worker thread.

        Raises
        ------
        RuntimeError
            If the executor has already been shut down. We reject rather
            than silently drop so callers notice lifecycle bugs.
        """
        if self._stopped.is_set():
            raise RuntimeError("SnapshotExecutor has been shut down; cannot submit new jobs")
        self._queue.put(job)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until every submitted job has completed (or *timeout*).

        Returns ``True`` if the queue drained in time, ``False`` on
        timeout. Uses a condition variable rather than polling so it is
        cheap to call in tests and shutdown paths.
        """
        deadline: float | None = None
        if timeout is not None:
            import time as _time
            deadline = _time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                if deadline is None:
                    self._queue.all_tasks_done.wait()
                else:
                    import time as _time
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
        # Sentinel wakes the worker even if the queue is empty.
        self._queue.put(None)
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # -- Worker loop ---------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                try:
                    job()
                except Exception:
                    # Isolate a bad job: one failure must not kill the
                    # worker and stop subsequent snapshots from persisting.
                    logger.exception("SnapshotExecutor job raised")
            finally:
                self._queue.task_done()
