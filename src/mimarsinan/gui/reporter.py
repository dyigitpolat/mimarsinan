"""GUI Reporter — captures metrics for the browser dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mimarsinan.gui.data_collector import DataCollector


class GUIReporter:
    """Drop-in replacement/companion for WandB_Reporter.

    Every ``report()`` call is forwarded to a :class:`DataCollector` which
    stores the time-series for the web frontend.
    """

    def __init__(self, collector: DataCollector) -> None:
        self._collector = collector
        self.prefix: str = ""

    def report(self, metric_name: str, metric_value: Any, step: int | None = None) -> None:
        self._collector.record_metric(metric_name, metric_value, step=step)

    def console_log(self, metric_name: str, metric_value: Any) -> None:
        pass
