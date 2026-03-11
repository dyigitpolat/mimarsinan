"""Reporter protocol and default in-tree implementation for pipeline metrics."""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Reporter(Protocol):
    """Protocol for pipeline metric reporting. Used by Pipeline and CompositeReporter."""

    prefix: str

    def report(self, metric_name: str, metric_value: Any, step: int | None = None) -> None: ...

    def console_log(self, metric_name: str, metric_value: Any) -> None: ...

    def finish(self) -> None: ...


class DefaultReporter:
    """In-tree reporter with throttled console output. Implements Reporter protocol."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self._report_timestamps: dict[str, float] = {}
        self._reporting_intervals: dict[str, float] = {}

    def console_log(self, metric_name: str, metric_value: Any) -> None:
        display_name = (self.prefix + " " + metric_name).strip() if self.prefix else metric_name
        if display_name not in self._report_timestamps:
            self._report_timestamps[display_name] = 0.0
        if display_name not in self._reporting_intervals:
            self._reporting_intervals[display_name] = 0.5

        current_timestamp = time.time()
        current_interval = current_timestamp - self._report_timestamps[display_name]

        if current_interval > self._reporting_intervals[display_name]:
            print(f"            {display_name}: {metric_value}")
            self._report_timestamps[display_name] = current_timestamp

            if current_interval < 5.0:
                self._reporting_intervals[display_name] *= 1.5
            if current_interval > 5.0:
                self._reporting_intervals[display_name] *= 0.75

    def report(self, metric_name: str, metric_value: Any, step: int | None = None) -> None:
        self.console_log(metric_name, metric_value)

    def finish(self) -> None:
        pass
