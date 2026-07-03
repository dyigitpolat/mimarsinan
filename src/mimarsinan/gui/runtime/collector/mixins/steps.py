"""Step lifecycle recording for DataCollector."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Iterable

from mimarsinan.gui.runtime.collector.types import MetricEvent, StepRecord, StepStatus

if TYPE_CHECKING:
    from mimarsinan.gui.resources import ResourceDescriptor

logger = logging.getLogger("mimarsinan.gui")


class StepsMixin:
    """Mixin: step lifecycle and persisted-step restore."""

    _lock: Any
    _step_names: list[str]
    _steps: dict[str, StepRecord]
    _metrics: list[MetricEvent]
    _metric_seq: int
    _current_step: str | None
    _resource_store: Any

    if TYPE_CHECKING:
        def _broadcast_lifecycle(self, message: dict) -> None: ...
        def _broadcast_pipeline_overview(self) -> None: ...

    def step_started(self, step_name: str) -> None:
        with self._lock:
            self._current_step = step_name
            rec = self._steps.get(step_name)
            if rec is None:
                rec = StepRecord(name=step_name)
                self._steps[step_name] = rec
                if step_name not in self._step_names:
                    self._step_names.append(step_name)
            rec.status = StepStatus.RUNNING
            rec.start_time = time.time()
            rec.end_time = None
            rec.target_metric = None
            rec.snapshot = None
            rec.snapshot_key_kinds = None
            rec.snapshot_version += 1
            self._metrics = [m for m in self._metrics if m.step_name != step_name]
            store = self._resource_store
        if store is not None:
            store.clear_step(step_name)
        self._broadcast_lifecycle({"type": "step_started", "step": step_name})
        self._broadcast_pipeline_overview()

    def step_completed(
        self,
        step_name: str,
        target_metric: float | None = None,
        snapshot: dict | None = None,
        snapshot_key_kinds: dict | None = None,
        resources: Iterable["ResourceDescriptor"] | None = None,
    ) -> None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is not None:
                rec.status = StepStatus.COMPLETED
                rec.end_time = time.time()
                rec.target_metric = target_metric
                if snapshot is not None:
                    rec.snapshot = snapshot
                if snapshot_key_kinds is not None:
                    rec.snapshot_key_kinds = snapshot_key_kinds
                rec.snapshot_version += 1
            if self._current_step == step_name:
                self._current_step = None
            store = self._resource_store
        if resources and store is not None:
            for desc in resources:
                store.put(step_name, desc)
        self._broadcast_lifecycle({
            "type": "step_completed",
            "step": step_name,
            "target_metric": target_metric,
        })
        self._broadcast_pipeline_overview()

    def step_failed(self, step_name: str, error: str = "") -> None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is not None:
                rec.status = StepStatus.FAILED
                rec.end_time = time.time()
                rec.error = error or None
                rec.snapshot_version += 1
            if self._current_step == step_name:
                self._current_step = None
        self._broadcast_lifecycle({"type": "step_failed", "step": step_name, "error": error})
        self._broadcast_pipeline_overview()

    def add_step_from_persisted(
        self,
        step_name: str,
        start_time: float,
        end_time: float,
        target_metric: float | None,
        metrics: list[dict],
        snapshot: dict | None,
        snapshot_key_kinds: dict | None = None,
    ) -> None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is None:
                rec = StepRecord(name=step_name)
                self._steps[step_name] = rec
                if step_name not in self._step_names:
                    self._step_names.append(step_name)
            rec.status = StepStatus.COMPLETED
            rec.start_time = start_time
            rec.end_time = end_time
            rec.target_metric = target_metric
            rec.snapshot = snapshot
            rec.snapshot_key_kinds = snapshot_key_kinds

            for m in metrics:
                seq = m.get("seq", 0)
                if seq > self._metric_seq:
                    self._metric_seq = seq
                self._metrics.append(MetricEvent(
                    seq=seq,
                    step_name=step_name,
                    metric_name=m.get("name", ""),
                    value=m.get("value"),
                    timestamp=m.get("timestamp", 0.0),
                    global_step=m.get("global_step"),
                ))
