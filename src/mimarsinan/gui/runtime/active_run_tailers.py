"""File tailers that push subprocess GUI state events."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from mimarsinan.common.best_effort import best_effort

logger = logging.getLogger("mimarsinan.gui.runtime.active_run_tailers")

POLL_INTERVAL_S = 0.05

Callback = Callable[[dict], None]


class JsonlTailer:
    """Tail a JSONL file and emit each record as a typed WS frame."""

    def __init__(self, path: Path, callback: Callback, *, frame_type: str) -> None:
        self._path = path
        self._callback = callback
        self._frame_type = frame_type
        self._stop = threading.Event()
        self._position = 0
        self._last_size = 0
        self._pending = b""
        self._thread = threading.Thread(
            target=self._run,
            name=f"JsonlTailer[{frame_type}:{path.parent.name}]",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            with best_effort(f"{self._frame_type} tailer tick for {self._path}", logger=logger):
                self._tick()
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
            with best_effort(f"{self._frame_type} tailer callback", logger=logger):
                self._callback({"type": self._frame_type, **record})


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
            with best_effort(f"steps watcher tick for {self._path}", logger=logger):
                self._tick()
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
        with best_effort("steps overview callback", logger=logger):
            self._callback({"type": "pipeline_overview", **overview})


def metrics_tailer(path: Path, callback: Callback) -> JsonlTailer:
    """Tailer for ``live_metrics.jsonl`` (``metric`` frames)."""
    return JsonlTailer(path, callback, frame_type="metric")


def events_tailer(path: Path, callback: Callback) -> JsonlTailer:
    """Tailer for ``events.jsonl`` (``event`` frames)."""
    return JsonlTailer(path, callback, frame_type="event")
