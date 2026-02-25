"""Mimarsinan Pipeline Monitoring GUI.

Usage::

    from mimarsinan.gui import start_gui

    gui = start_gui(pipeline, port=8501)
    pipeline.register_pre_step_hook(gui.on_step_start)
    pipeline.register_post_step_hook(gui.on_step_end)

The ``GUIHandle`` returned by :func:`start_gui` provides hook callbacks
and exposes the underlying :class:`DataCollector` and :class:`GUIReporter`.
"""

from __future__ import annotations

from typing import Any

from mimarsinan.gui.data_collector import DataCollector
from mimarsinan.gui.reporter import GUIReporter
from mimarsinan.gui.snapshot import build_step_snapshot


class GUIHandle:
    """Facade returned by :func:`start_gui`."""

    def __init__(self, pipeline: Any, collector: DataCollector) -> None:
        self.pipeline = pipeline
        self.collector = collector
        self.reporter = GUIReporter(collector)

    def on_step_start(self, step_name: str, step: Any) -> None:
        self.reporter.prefix = step_name
        self.collector.step_started(step_name)

    def on_step_end(self, step_name: str, step: Any) -> None:
        try:
            raw = self.pipeline.get_target_metric()
            target_metric = float(raw) if raw is not None else None
        except Exception:
            target_metric = None
        try:
            snapshot = build_step_snapshot(self.pipeline, step_name)
        except Exception:
            snapshot = None
        self.collector.step_completed(step_name, target_metric=target_metric, snapshot=snapshot)


def start_gui(pipeline: Any, *, port: int = 8501, host: str = "0.0.0.0") -> GUIHandle:
    """Spin up the GUI server and return a handle for hook registration."""
    from mimarsinan.gui.server import start_server

    collector = DataCollector()

    step_names = [name for name, _ in pipeline.steps]
    config = getattr(pipeline, "config", {})
    safe_config = _make_json_safe(config)
    collector.set_pipeline_info(step_names, safe_config)

    start_server(collector, host=host, port=port)
    return GUIHandle(pipeline, collector)


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
