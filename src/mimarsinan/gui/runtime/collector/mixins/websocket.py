"""WebSocket listener management and broadcast for DataCollector."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from typing import Any

logger = logging.getLogger("mimarsinan.gui")


class WebSocketMixin:
    """Mixin: WS listeners, lifecycle buffering, and broadcast."""

    _lock: Any
    _ws_listeners: list[Any]
    _event_seq: int
    _event_buffer: deque[dict]

    def add_ws_listener(self, ws: Any) -> None:
        with self._lock:
            self._ws_listeners.append(ws)

    def remove_ws_listener(self, ws: Any) -> None:
        with self._lock:
            if ws in self._ws_listeners:
                self._ws_listeners.remove(ws)

    def _broadcast_lifecycle(self, message: dict) -> None:
        with self._lock:
            self._event_seq += 1
            tagged = dict(message)
            tagged["event_seq"] = self._event_seq
            self._event_buffer.append(tagged)
        self._broadcast(tagged)

    def get_current_event_seq(self) -> int:
        with self._lock:
            return self._event_seq

    def replay_events_since(self, ws: Any, last_seq: int) -> None:
        with self._lock:
            pending = [e for e in self._event_buffer if e.get("event_seq", 0) > last_seq]
        loop = ws._loop if hasattr(ws, "_loop") else None
        if loop is None or not loop.is_running():
            return
        for evt in pending:
            try:
                fut = asyncio.run_coroutine_threadsafe(ws.send_json(evt), loop)
                fut.result(timeout=2.0)
            except Exception:
                logger.warning("WS resume replay failed; aborting", exc_info=True)
                return
        try:
            overview = self.get_pipeline_overview()
        except Exception:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "pipeline_overview", **overview}), loop,
            )
            fut.result(timeout=2.0)
        except Exception:
            logger.warning("WS resume overview push failed", exc_info=True)

    def _broadcast(self, message: dict) -> None:
        def _ws_sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: _ws_sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_ws_sanitize(v) for v in obj]
            return obj

        safe_message = _ws_sanitize(message)

        with self._lock:
            listeners = list(self._ws_listeners)
        dead: list[Any] = []
        for ws in listeners:
            try:
                loop = ws._loop if hasattr(ws, "_loop") else None
                if loop is not None and loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(
                        ws.send_json(safe_message), loop,
                    )
                    try:
                        fut.result(timeout=2.0)
                    except Exception:
                        logger.warning(
                            "WebSocket send timed out or failed; dropping listener",
                            exc_info=True,
                        )
                        dead.append(ws)
                else:
                    dead.append(ws)
            except Exception:
                logger.warning(
                    "Failed to broadcast to WebSocket, removing listener",
                    exc_info=True,
                )
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    if ws in self._ws_listeners:
                        self._ws_listeners.remove(ws)
