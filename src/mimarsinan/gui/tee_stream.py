"""Tee stdout/stderr into the GUI console log collector."""

from __future__ import annotations

import io
import threading
from typing import Any


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
        try:
            self._original.write(s)
            self._original.flush()
        except Exception:
            pass
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                try:
                    self._callback(line)
                except Exception:
                    pass
        return len(s)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def flush_remaining(self) -> None:
        with self._lock:
            if self._buf:
                try:
                    self._callback(self._buf)
                except Exception:
                    pass
                self._buf = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)
