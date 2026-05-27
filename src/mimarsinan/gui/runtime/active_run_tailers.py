"""File tailers that push subprocess GUI state events."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("mimarsinan.gui.runtime.active_run_tailers")

POLL_INTERVAL_S = 0.05

Callback = Callable[[dict], None]


class MetricsTailer:
    """Tail ``live_metrics.jsonl`` and emit per-line metric events."""

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
            self._stop.wait(POLL_INTERVAL_S)

    def _tick(self) -> None:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return
        except OSError:
            return
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


class StepsFileWatcher:
    """Re-read ``steps.json`` on mtime change and emit pipeline-overview events."""

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
            self._stop.wait(POLL_INTERVAL_S)

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
