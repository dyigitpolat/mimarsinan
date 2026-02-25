"""Composite reporter that dispatches to multiple backends."""

from __future__ import annotations

from typing import Any, Sequence


class CompositeReporter:
    """Broadcasts ``report()`` and ``console_log()`` to every child reporter.

    Behaves like a single reporter from the pipeline's perspective.
    """

    def __init__(self, reporters: Sequence[Any]) -> None:
        self._reporters = list(reporters)
        self._prefix: str = ""

    @property
    def prefix(self) -> str:
        return self._prefix

    @prefix.setter
    def prefix(self, value: str) -> None:
        self._prefix = value
        for r in self._reporters:
            r.prefix = value

    def report(self, metric_name: str, metric_value: Any, step: int | None = None) -> None:
        for r in self._reporters:
            r.report(metric_name, metric_value, step)

    def console_log(self, metric_name: str, metric_value: Any) -> None:
        for r in self._reporters:
            r.console_log(metric_name, metric_value)
