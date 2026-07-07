"""Shared types and helpers for the GUI data collector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mimarsinan.gui.json_util import to_json_safe as _to_json_safe


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
class ConsoleLogEntry:
    seq: int
    stream: str
    line: str
    ts: float


@dataclass
class StepRecord:
    name: str
    status: StepStatus = StepStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    target_metric: float | None = None
    snapshot: dict | None = None
    snapshot_key_kinds: dict | None = None
    error: str | None = None
    snapshot_version: int = 0
    metric_kind: str | None = None
    verdict: dict | None = None


def build_snapshot_etag(rec: StepRecord) -> str:
    """Weak HTTP ETag for a step record."""
    return f'W/"{rec.name}-{rec.status.value}-{rec.snapshot_version}"'


def to_json_safe(value: Any) -> Any:
    return _to_json_safe(value)
