"""Metric recording for DataCollector."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.runtime.collector.types import MetricEvent, to_json_safe

logger = logging.getLogger("mimarsinan.gui")


class MetricsMixin:
    """Mixin: metric time-series append and broadcast."""

    _lock: Any
    _metrics: list[MetricEvent]
    _metric_seq: int
    _current_step: str | None
    _metric_callback: Any

    if TYPE_CHECKING:
        def _broadcast(self, message: dict) -> None: ...

    def record_metric(self, metric_name: str, value: Any, step: int | None = None) -> None:
        with self._lock:
            self._metric_seq += 1
            current = self._current_step or ""
            evt = MetricEvent(
                seq=self._metric_seq,
                step_name=current,
                metric_name=metric_name,
                value=to_json_safe(value),
                timestamp=time.time(),
                global_step=step,
            )
            self._metrics.append(evt)
            cb = self._metric_callback
        self._broadcast({
            "type": "metric",
            "step": current,
            "name": metric_name,
            "value": evt.value,
            "seq": evt.seq,
            "timestamp": evt.timestamp,
        })
        if cb is not None:
            with best_effort("metric callback", logger=logger):
                cb(current, metric_name, evt.value, evt.seq, evt.timestamp)

    def get_step_metrics(self, step_name: str) -> list[dict]:
        with self._lock:
            return [
                {
                    "seq": m.seq,
                    "name": m.metric_name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "global_step": m.global_step,
                }
                for m in self._metrics
                if m.step_name == step_name
            ]

    def get_all_metrics(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "seq": m.seq,
                    "step": m.step_name,
                    "name": m.metric_name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                }
                for m in self._metrics
            ]
