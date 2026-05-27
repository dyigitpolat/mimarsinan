"""Subscription hub for active-run WebSocket streams."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from mimarsinan.gui.runtime.active_run_tailers import MetricsTailer, StepsFileWatcher
from mimarsinan.gui.runtime.persistence.paths import (
    GUI_STATE_DIR,
    LIVE_METRICS_FILENAME,
    STEPS_FILENAME,
)

logger = logging.getLogger("mimarsinan.gui.runtime.active_run_hub")


class _RunSubscribers:
    def __init__(self, metrics: MetricsTailer, steps: StepsFileWatcher) -> None:
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
    """Manages per-run tailers and broadcasts their events to WS subscribers."""

    def __init__(
        self,
        get_working_dir: Callable[[str], Optional[str]],
        build_overview: Callable[[str], Optional[dict]],
    ) -> None:
        self._get_working_dir = get_working_dir
        self._build_overview = build_overview
        self._runs: dict[str, _RunSubscribers] = {}
        self._lock = threading.Lock()
        self._send_fns: dict[Any, Callable[[dict], None]] = {}

    def subscribe(
        self, run_id: str, subscriber: Any, send_fn: Callable[[dict], None],
    ) -> bool:
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
            state_dir = Path(working_dir) / GUI_STATE_DIR
            metrics_path = state_dir / LIVE_METRICS_FILENAME
            steps_path = state_dir / STEPS_FILENAME

            def on_metric(msg: dict) -> None:
                self._broadcast(run_id, msg)

            def on_overview(msg: dict) -> None:
                self._broadcast(run_id, msg)

            metrics_tailer = MetricsTailer(metrics_path, on_metric)
            steps_watcher = StepsFileWatcher(
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
            self._runs.pop(run_id, None)
        try:
            record.metrics.stop()
            record.steps.stop()
        except Exception:
            logger.debug("Failed to stop tailers for run %s", run_id, exc_info=True)

    def shutdown(self) -> None:
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
