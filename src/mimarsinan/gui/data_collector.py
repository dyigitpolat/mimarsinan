"""Central event store for the GUI monitoring system.

Accumulates metric time-series, step lifecycle events, and rich snapshot
data.  Thread-safe so the FastAPI server can read while the pipeline
writes from its own thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("mimarsinan.gui")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MetricEvent:
    seq: int
    step_name: str
    metric_name: str
    value: Any
    timestamp: float
    global_step: int | None = None


@dataclass
class StepRecord:
    name: str
    status: StepStatus = StepStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    target_metric: float | None = None
    snapshot: dict | None = None


class DataCollector:
    """Centralized, thread-safe store for pipeline monitoring data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._step_names: list[str] = []
        self._steps: dict[str, StepRecord] = {}
        self._metrics: list[MetricEvent] = []
        self._metric_seq: int = 0
        self._current_step: str | None = None
        self._pipeline_config: dict | None = None

        self._ws_listeners: list[Any] = []

    # -- Pipeline configuration ------------------------------------------------

    def set_pipeline_info(self, step_names: list[str], config: dict) -> None:
        with self._lock:
            self._step_names = list(step_names)
            self._pipeline_config = dict(config)
            self._steps = {
                name: StepRecord(name=name) for name in step_names
            }

    @property
    def pipeline_config(self) -> dict | None:
        with self._lock:
            return self._pipeline_config

    # -- Step lifecycle --------------------------------------------------------

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
        self._broadcast({"type": "step_started", "step": step_name})

    def step_completed(
        self, step_name: str, target_metric: float | None = None, snapshot: dict | None = None
    ) -> None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is not None:
                rec.status = StepStatus.COMPLETED
                rec.end_time = time.time()
                rec.target_metric = target_metric
                if snapshot is not None:
                    rec.snapshot = snapshot
            if self._current_step == step_name:
                self._current_step = None
        self._broadcast({
            "type": "step_completed",
            "step": step_name,
            "target_metric": target_metric,
        })

    def step_failed(self, step_name: str, error: str = "") -> None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is not None:
                rec.status = StepStatus.FAILED
                rec.end_time = time.time()
            if self._current_step == step_name:
                self._current_step = None
        self._broadcast({"type": "step_failed", "step": step_name, "error": error})

    # -- Metrics ---------------------------------------------------------------

    def record_metric(self, metric_name: str, value: Any, step: int | None = None) -> None:
        with self._lock:
            self._metric_seq += 1
            current = self._current_step or ""
            evt = MetricEvent(
                seq=self._metric_seq,
                step_name=current,
                metric_name=metric_name,
                value=_to_json_safe(value),
                timestamp=time.time(),
                global_step=step,
            )
            self._metrics.append(evt)
        self._broadcast({
            "type": "metric",
            "step": current,
            "name": metric_name,
            "value": evt.value,
            "seq": evt.seq,
            "timestamp": evt.timestamp,
        })

    # -- Read API (called by the server) --------------------------------------

    def get_pipeline_overview(self) -> dict:
        with self._lock:
            steps = []
            for name in self._step_names:
                rec = self._steps.get(name, StepRecord(name=name))
                steps.append({
                    "name": rec.name,
                    "status": rec.status.value,
                    "start_time": rec.start_time,
                    "end_time": rec.end_time,
                    "duration": (rec.end_time - rec.start_time) if rec.start_time and rec.end_time else None,
                    "target_metric": rec.target_metric,
                })
            return {
                "steps": steps,
                "current_step": self._current_step,
                "config": self._pipeline_config,
            }

    def get_step_detail(self, step_name: str) -> dict | None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is None:
                return None
            metrics = [
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
            return {
                "name": rec.name,
                "status": rec.status.value,
                "start_time": rec.start_time,
                "end_time": rec.end_time,
                "duration": (rec.end_time - rec.start_time) if rec.start_time and rec.end_time else None,
                "target_metric": rec.target_metric,
                "metrics": metrics,
                "snapshot": rec.snapshot,
            }

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

    # -- WebSocket listener management ----------------------------------------

    def add_ws_listener(self, ws: Any) -> None:
        with self._lock:
            self._ws_listeners.append(ws)

    def remove_ws_listener(self, ws: Any) -> None:
        with self._lock:
            if ws in self._ws_listeners:
                self._ws_listeners.remove(ws)

    def _broadcast(self, message: dict) -> None:
        import asyncio

        with self._lock:
            listeners = list(self._ws_listeners)
        dead: list[Any] = []
        for ws in listeners:
            try:
                loop = ws._loop if hasattr(ws, "_loop") else None
                if loop is not None and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        ws.send_json(message), loop
                    )
                else:
                    dead.append(ws)
            except Exception:
                logger.debug("Failed to broadcast to WebSocket, removing listener", exc_info=True)
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    if ws in self._ws_listeners:
                        self._ws_listeners.remove(ws)


def _to_json_safe(value: Any) -> Any:
    """Convert tensors / numpy values to plain Python types."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        float(value)
        return float(value)
    except (TypeError, ValueError):
        return str(value)
