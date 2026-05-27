"""Central thread-safe event store for the GUI monitoring system."""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

from mimarsinan.gui.runtime.collector.mixins import (
    ConsoleMixin,
    MetricsMixin,
    ReadApiMixin,
    StepsMixin,
    WebSocketMixin,
)
from mimarsinan.gui.runtime.collector.types import StepRecord

if TYPE_CHECKING:
    from mimarsinan.gui.resources import ResourceStore


class DataCollector(
    StepsMixin,
    MetricsMixin,
    ConsoleMixin,
    ReadApiMixin,
    WebSocketMixin,
):
    """Centralized, thread-safe store for pipeline monitoring data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._step_names: list[str] = []
        self._steps: dict[str, StepRecord] = {}
        self._metrics: list = []
        self._metric_seq: int = 0
        self._current_step: str | None = None
        self._pipeline_config: dict | None = None

        self._console_logs: list = []
        self._console_seq: int = 0

        self._ws_listeners: list[Any] = []
        self._pipeline_thread: Optional[threading.Thread] = None
        self._metric_callback: Any = None
        self._console_callback: Any = None

        self._resource_store: Optional["ResourceStore"] = None
        self._working_directory: Optional[str] = None

        self._event_seq: int = 0
        self._event_buffer: deque[dict] = deque(maxlen=512)

    def set_resource_store(self, store: "ResourceStore | None") -> None:
        with self._lock:
            self._resource_store = store

    def get_resource_store(self) -> "ResourceStore | None":
        with self._lock:
            return self._resource_store

    def set_working_directory(self, working_dir: Optional[str]) -> None:
        with self._lock:
            self._working_directory = working_dir

    def get_working_directory(self) -> Optional[str]:
        with self._lock:
            return self._working_directory

    def set_pipeline_thread(self, thread: Optional[threading.Thread]) -> None:
        with self._lock:
            self._pipeline_thread = thread

    def get_pipeline_thread(self) -> Optional[threading.Thread]:
        with self._lock:
            return self._pipeline_thread

    def join_pipeline_thread(self, timeout: float = 30.0) -> bool:
        with self._lock:
            t = self._pipeline_thread
        if t is None:
            return True
        t.join(timeout=timeout)
        return not t.is_alive()

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
