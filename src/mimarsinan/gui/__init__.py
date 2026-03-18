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
from mimarsinan.gui.persistence import (
    load_persisted_steps,
    save_step_to_persisted,
    append_live_metric,
)
from mimarsinan.gui.reporter import GUIReporter
from mimarsinan.gui.snapshot import build_step_snapshot


class GUIHandle:
    """Facade returned by :func:`start_gui`."""

    def __init__(
        self, pipeline: Any, collector: DataCollector, persist_metrics: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.collector = collector
        self.reporter = GUIReporter(collector)
        self._persist_metrics = persist_metrics

    def on_step_start(self, step_name: str, step: Any) -> None:
        self.reporter.prefix = step_name
        self.collector.step_started(step_name)
        working_dir = getattr(self.pipeline, "working_directory", None)
        if working_dir:
            import time
            save_step_to_persisted(
                working_dir, step_name,
                start_time=time.time(), end_time=None,
                target_metric=None, metrics=[], snapshot=None,
                snapshot_key_kinds=None, status="running",
            )

    def on_metric(
        self, step_name: str, metric_name: str, value: float, seq: int, timestamp: float,
    ) -> None:
        if not self._persist_metrics:
            return
        working_dir = getattr(self.pipeline, "working_directory", None)
        if working_dir:
            append_live_metric(working_dir, step_name, metric_name, value, seq, timestamp)

    def on_step_end(self, step_name: str, step: Any) -> None:
        try:
            raw = self.pipeline.get_target_metric()
            target_metric = float(raw) if raw is not None else None
        except Exception:
            target_metric = None
        try:
            snapshot, snapshot_key_kinds = build_step_snapshot(
                self.pipeline, step_name, step=step
            )
        except Exception as e:
            self.collector.step_failed(step_name, error=str(e))
            return
        self.collector.step_completed(
            step_name,
            target_metric=target_metric,
            snapshot=snapshot,
            snapshot_key_kinds=snapshot_key_kinds,
        )
        # Persist this step so it can be restored when starting from a later step
        working_dir = getattr(self.pipeline, "working_directory", None)
        if working_dir:
            detail = self.collector.get_step_detail(step_name)
            if detail:
                save_step_to_persisted(
                    working_dir,
                    step_name,
                    detail.get("start_time"),
                    detail.get("end_time"),
                    detail.get("target_metric"),
                    detail.get("metrics", []),
                    detail.get("snapshot"),
                    detail.get("snapshot_key_kinds"),
                    status="completed",
                )


def start_gui(
    pipeline: Any,
    *,
    port: int = 8501,
    host: str = "0.0.0.0",
    start_step: str | None = None,
) -> GUIHandle:
    """Spin up the GUI server and return a handle for hook registration.

    If start_step is set, steps before it are backfilled from the pipeline cache
    so they can be browsed (metrics, Model, IR Graph, Hardware tabs) even though
    they did not run in this session.
    """
    from mimarsinan.gui.server import start_server

    collector = DataCollector()

    step_names = [name for name, _ in pipeline.steps]
    config = getattr(pipeline, "config", {})
    safe_config = _make_json_safe(config)
    collector.set_pipeline_info(step_names, safe_config)

    if start_step is not None:
        _backfill_skipped_steps(pipeline, collector, step_names, start_step)

    start_server(collector, host=host, port=port)
    return GUIHandle(pipeline, collector)


def _backfill_skipped_steps(
    pipeline: Any,
    collector: DataCollector,
    step_names: list[str],
    start_step: str,
) -> None:
    """Restore steps before start_step from persisted state or cache (step-specific snapshot)."""
    try:
        start_idx = step_names.index(start_step)
    except ValueError:
        return
    working_dir = getattr(pipeline, "working_directory", "")
    persisted = load_persisted_steps(working_dir) if working_dir else {}

    step_by_name = {name: step for name, step in pipeline.steps}

    for i in range(start_idx):
        step_name = step_names[i]
        data = persisted.get(step_name)
        if data is not None:
            collector.add_step_from_persisted(
                step_name,
                data.get("start_time", 0.0),
                data.get("end_time", 0.0),
                data.get("target_metric"),
                data.get("metrics", []),
                data.get("snapshot"),
                data.get("snapshot_key_kinds"),
            )
        else:
            step = step_by_name.get(step_name)
            try:
                snapshot, snapshot_key_kinds = build_step_snapshot(
                    pipeline, step_name, step=step
                )
            except Exception:
                snapshot = None
                snapshot_key_kinds = None
            collector.step_completed(
                step_name,
                target_metric=None,
                snapshot=snapshot,
                snapshot_key_kinds=snapshot_key_kinds,
            )


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
