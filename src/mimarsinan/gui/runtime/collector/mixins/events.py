"""Structured pipeline-event recording for DataCollector."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.runtime.events import PipelineEvent
from mimarsinan.gui.runtime.collector.types import to_json_safe

logger = logging.getLogger("mimarsinan.gui")


class EventsMixin:
    """Mixin: structured event append, broadcast, and persistence callback."""

    _lock: Any
    _pipeline_events: list[PipelineEvent]
    _pipeline_event_seq: int
    _current_step: str | None
    _pipeline_event_callback: Any

    if TYPE_CHECKING:
        def _broadcast(self, message: dict) -> None: ...

    def record_event(self, kind: str, payload: dict) -> None:
        with self._lock:
            self._pipeline_event_seq += 1
            event = PipelineEvent(
                seq=self._pipeline_event_seq,
                step_name=self._current_step or "",
                kind=str(kind),
                payload=to_json_safe(payload or {}),
                timestamp=time.time(),
            )
            self._pipeline_events.append(event)
            callback = self._pipeline_event_callback
        self._broadcast({"type": "event", **event.to_record()})
        if callback is not None:
            with best_effort("pipeline event callback", logger=logger):
                callback(event)

    def set_event_callback(self, callback: Any) -> None:
        with self._lock:
            self._pipeline_event_callback = callback

    def get_events(self, *, since_seq: int = 0, step_name: str | None = None) -> list[dict]:
        with self._lock:
            return [
                e.to_record()
                for e in self._pipeline_events
                if e.seq > since_seq and (step_name is None or e.step_name == step_name)
            ]
