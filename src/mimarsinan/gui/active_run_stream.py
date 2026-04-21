"""Real-time metric/lifecycle streaming for subprocess-spawned active runs.

Background
----------
When a pipeline runs under ``--ui`` (via :class:`ProcessManager`), the GUI
server monitors a child process through its ``_GUI_STATE/`` files
(``steps.json``, ``live_metrics.jsonl``). The frontend previously polled
those files every 3 seconds, which made the charts update in coarse
batches: smooth real-time plots need per-metric push, not a periodic
flush.

This module sits between the subprocess's file-based IPC and the
parent's WebSocket clients. For every *active* run we tail:

* ``live_metrics.jsonl`` — one new JSON line per metric append.
* ``steps.json``         — re-read whenever its mtime changes, yielding
  ``pipeline_overview``-shaped messages so the pipeline bar updates as
  soon as a step transitions.

Subscribers register via :meth:`ActiveRunHub.subscribe`. The hub
reference-counts tailers per run so a single run is tailed exactly once
regardless of how many browser tabs are open. When the last subscriber
disconnects the tailer is stopped.

Design choices
--------------
* Poll-based (50 ms) rather than inotify. Inotify would be native on
  Linux but cross-platform and racey with atomic-rename writes
  (``steps.json`` is written via ``tmp + rename``, so inotify IN_CLOSE
  semantics differ across OSes). The 50 ms poll is cheap: ``os.stat`` +
  seek-to-EOF reads, with a tracked byte offset for the append-only log.
* Single dedicated thread per tailed file; the hub joins both on stop.
* Per-run locks protect subscriber membership; no global lock is held
  during callback dispatch so slow WS clients cannot stall the tailer.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from mimarsinan.gui.persistence import (
    _GUI_STATE_DIR,
    _LIVE_METRICS_FILENAME,
    _STEPS_FILENAME,
)

logger = logging.getLogger("mimarsinan.gui.active_run_stream")

_POLL_INTERVAL_S = 0.05  # 50 ms — target ~20 Hz UI update cadence.


Callback = Callable[[dict], None]


class _MetricsTailer:
    """Tails a subprocess's ``live_metrics.jsonl`` and emits per-line events.

    Each line is a JSON record ``{step, name, value, seq, timestamp}``
    appended by :func:`append_live_metric`. We track a byte offset so
    we only read new data per tick.
    """

    def __init__(self, path: Path, callback: Callback) -> None:
        self._path = path
        self._callback = callback
        self._stop = threading.Event()
        self._position = 0
        self._last_size = 0
        self._pending = b""
        self._thread = threading.Thread(
            target=self._run, name=f"MetricsTailer[{path.parent.name}]", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.debug("Metrics tailer tick failed for %s", self._path, exc_info=True)
            self._stop.wait(_POLL_INTERVAL_S)

    def _tick(self) -> None:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return
        except OSError:
            return
        # Truncation detection: the file shrank since we last looked.
        # Compare against ``_last_size`` rather than ``_position`` because
        # after a truncate+append cycle the new size can coincide with
        # the old position, leaving ``size < position`` false while the
        # payload we expect to tail is actually *different* content.
        if size < self._last_size:
            self._position = 0
            self._pending = b""
        self._last_size = size
        if size <= self._position:
            return
        try:
            with open(self._path, "rb") as f:
                f.seek(self._position)
                chunk = f.read(size - self._position)
        except OSError:
            return
        self._position += len(chunk)
        self._pending += chunk
        while b"\n" in self._pending:
            line, self._pending = self._pending.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            try:
                self._callback({"type": "metric", **record})
            except Exception:
                logger.debug("Metrics callback raised", exc_info=True)


class _StepsFileWatcher:
    """Re-reads ``steps.json`` on mtime change and emits pipeline-overview events.

    This lets the pipeline bar, overview cards and step badges update as
    soon as a step transitions, without the frontend having to poll.
    """

    def __init__(
        self,
        path: Path,
        build_overview: Callable[[], Optional[dict]],
        callback: Callback,
    ) -> None:
        self._path = path
        self._build_overview = build_overview
        self._callback = callback
        self._stop = threading.Event()
        self._last_mtime_ns: int = 0
        self._thread = threading.Thread(
            target=self._run, name=f"StepsWatcher[{path.parent.name}]", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.debug("Steps watcher tick failed for %s", self._path, exc_info=True)
            self._stop.wait(_POLL_INTERVAL_S)

    def _tick(self) -> None:
        try:
            mtime_ns = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        except OSError:
            return
        if mtime_ns == self._last_mtime_ns:
            return
        self._last_mtime_ns = mtime_ns
        overview = self._build_overview()
        if overview is None:
            return
        try:
            self._callback({"type": "pipeline_overview", **overview})
        except Exception:
            logger.debug("Steps overview callback raised", exc_info=True)


class _RunSubscribers:
    """Per-run subscriber set + its tailers. Ref-counted by the hub."""

    def __init__(self, metrics: _MetricsTailer, steps: _StepsFileWatcher) -> None:
        self.metrics = metrics
        self.steps = steps
        self.subscribers: set[Any] = set()
        self.lock = threading.Lock()

    def add(self, sub: Any) -> int:
        with self.lock:
            self.subscribers.add(sub)
            return len(self.subscribers)

    def remove(self, sub: Any) -> int:
        with self.lock:
            self.subscribers.discard(sub)
            return len(self.subscribers)

    def snapshot(self) -> list[Any]:
        with self.lock:
            return list(self.subscribers)


class ActiveRunHub:
    """Manages per-run tailers and broadcasts their events to WS subscribers.

    Thread model
    ------------
    * Tailer threads produce events.
    * Broadcast runs on the tailer thread but dispatches to each
      subscriber through a user-supplied ``send_fn``. The ``send_fn`` is
      expected to schedule the actual network write on the asyncio loop
      so the tailer never blocks on slow clients.
    """

    def __init__(
        self,
        get_working_dir: Callable[[str], Optional[str]],
        build_overview: Callable[[str], Optional[dict]],
    ) -> None:
        self._get_working_dir = get_working_dir
        self._build_overview = build_overview
        self._runs: dict[str, _RunSubscribers] = {}
        self._lock = threading.Lock()
        # Subscriber → send_fn registry. The send_fn is called with a
        # sanitized JSON dict; failures remove the subscriber.
        self._send_fns: dict[Any, Callable[[dict], None]] = {}

    def subscribe(
        self, run_id: str, subscriber: Any, send_fn: Callable[[dict], None],
    ) -> bool:
        """Register *subscriber* for *run_id*'s event stream.

        Returns ``True`` if the subscription was established, ``False``
        if the run has no working directory (unknown run).
        """
        with self._lock:
            self._send_fns[subscriber] = send_fn
            existing = self._runs.get(run_id)
            if existing is not None:
                existing.add(subscriber)
                return True
            working_dir = self._get_working_dir(run_id)
            if not working_dir:
                self._send_fns.pop(subscriber, None)
                return False
            state_dir = Path(working_dir) / _GUI_STATE_DIR
            metrics_path = state_dir / _LIVE_METRICS_FILENAME
            steps_path = state_dir / _STEPS_FILENAME

            # Capture run_id for the tailer callbacks.
            def on_metric(msg: dict) -> None:
                self._broadcast(run_id, msg)

            def on_overview(msg: dict) -> None:
                self._broadcast(run_id, msg)

            metrics_tailer = _MetricsTailer(metrics_path, on_metric)
            steps_watcher = _StepsFileWatcher(
                steps_path,
                build_overview=lambda rid=run_id: self._build_overview(rid),
                callback=on_overview,
            )
            record = _RunSubscribers(metrics_tailer, steps_watcher)
            record.add(subscriber)
            self._runs[run_id] = record
            metrics_tailer.start()
            steps_watcher.start()
            return True

    def unsubscribe(self, run_id: str, subscriber: Any) -> None:
        with self._lock:
            self._send_fns.pop(subscriber, None)
            record = self._runs.get(run_id)
            if record is None:
                return
            remaining = record.remove(subscriber)
            if remaining > 0:
                return
            # Last subscriber gone → stop tailers.
            self._runs.pop(run_id, None)
        # Stop outside the lock to avoid holding it during join().
        try:
            record.metrics.stop()
            record.steps.stop()
        except Exception:
            logger.debug("Failed to stop tailers for run %s", run_id, exc_info=True)

    def shutdown(self) -> None:
        """Stop every tailer. Used on server shutdown."""
        with self._lock:
            records = list(self._runs.values())
            self._runs.clear()
            self._send_fns.clear()
        for rec in records:
            try:
                rec.metrics.stop()
                rec.steps.stop()
            except Exception:
                logger.debug("Failed to stop tailers on shutdown", exc_info=True)

    def _broadcast(self, run_id: str, message: dict) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            subs = record.snapshot()
            send_fns = {s: self._send_fns.get(s) for s in subs}
        for sub in subs:
            fn = send_fns.get(sub)
            if fn is None:
                continue
            try:
                fn(message)
            except Exception:
                logger.debug("Subscriber send_fn raised", exc_info=True)


__all__ = ["ActiveRunHub"]
