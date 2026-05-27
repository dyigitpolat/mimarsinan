"""Console log recording for DataCollector."""

from __future__ import annotations

import time
from typing import Any

from mimarsinan.gui.runtime.collector.types import ConsoleLogEntry


class ConsoleMixin:
    """Mixin: stdout/stderr line capture."""

    _lock: Any
    _console_logs: list[ConsoleLogEntry]
    _console_seq: int
    _console_callback: Any

    def record_console_log(self, line: str, stream: str) -> None:
        with self._lock:
            self._console_seq += 1
            entry = ConsoleLogEntry(
                seq=self._console_seq,
                stream=stream,
                line=line,
                ts=time.time(),
            )
            self._console_logs.append(entry)
            cb = self._console_callback
        self._broadcast({
            "type": "console_log",
            "stream": stream,
            "line": line,
            "ts": entry.ts,
            "seq": entry.seq,
        })
        if cb is not None:
            try:
                cb(stream, line, entry.ts)
            except Exception:
                pass

    def get_console_logs(self, offset: int = 0) -> list[dict]:
        with self._lock:
            return [
                {"seq": e.seq, "stream": e.stream, "line": e.line, "ts": e.ts}
                for e in self._console_logs[offset:]
            ]
