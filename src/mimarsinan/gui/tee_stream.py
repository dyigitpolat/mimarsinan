"""Tee stdout/stderr into the GUI console log collector."""

from __future__ import annotations

import io
import logging
import threading
from typing import Any

from mimarsinan.common.best_effort import best_effort

logger = logging.getLogger("mimarsinan.gui")


class TeeStream(io.RawIOBase):
    """Forward complete lines to a callback."""

    def __init__(self, original: Any, callback: Any) -> None:
        self._original = original
        self._callback = callback
        self._buf = ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, s: Any) -> int:  # type: ignore[override]
        if isinstance(s, (bytes, bytearray)):
            text = s.decode("utf-8", errors="replace")
        else:
            text = str(s)
        with best_effort("tee write-through to original stream", logger=logger):
            self._original.write(s)
            self._original.flush()
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                with best_effort("tee console callback", logger=logger):
                    self._callback(line)
        return len(s)

    def flush(self) -> None:
        with best_effort("tee flush original stream", logger=logger):
            self._original.flush()

    def flush_remaining(self) -> None:
        with self._lock:
            if self._buf:
                with best_effort("tee flush remaining console callback", logger=logger):
                    self._callback(self._buf)
                self._buf = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)
