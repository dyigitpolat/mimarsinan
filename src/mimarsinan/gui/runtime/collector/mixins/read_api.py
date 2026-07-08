"""Read-side API for DataCollector (REST / overview)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.runtime.collector.types import StepRecord, build_snapshot_etag
from mimarsinan.gui.viewmodel import (
    annotations_for_step,
    build_overview_chart,
    categories_for,
    step_bar_badge,
)

logger = logging.getLogger("mimarsinan.gui")


class ReadApiMixin:
    """Mixin: pipeline overview and step detail queries."""

    _lock: Any
    _step_names: list[str]
    _steps: dict[str, StepRecord]
    _metrics: list
    _pipeline_config: dict | None
    _current_step: str | None
    _working_directory: str | None

    if TYPE_CHECKING:
        def _broadcast(self, message: dict) -> None: ...

    def get_pipeline_overview(self) -> dict:
        with self._lock:
            config = self._pipeline_config or {}
            working_dir = self._working_directory
        groups: dict = {}
        with best_effort("build semantic groups for pipeline overview", logger=logger):
            from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
                get_pipeline_semantic_group_by_step_name,
            )
            groups = get_pipeline_semantic_group_by_step_name(config)
        config_view = None
        if config:
            with best_effort("build config_view for pipeline overview", logger=logger):
                from mimarsinan.config_schema.display_view import build_pipeline_config_view
                config_view = build_pipeline_config_view(config, working_dir=working_dir)
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
                    "metric_kind": rec.metric_kind,
                    "verdict": rec.verdict,
                    "semantic_group": groups.get(rec.name),
                })
            for step in steps:
                step["badge"] = step_bar_badge(step)
            overview = {
                "steps": steps,
                "current_step": self._current_step,
                "config": self._pipeline_config,
                "overview_chart": build_overview_chart(steps),
            }
            if config_view is not None:
                overview["config_view"] = config_view
            return overview

    def get_step_detail(
        self,
        step_name: str,
        *,
        since_seq: int = 0,
    ) -> dict | None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is None:
                return None
            step_metrics = [m for m in self._metrics if m.step_name == step_name]
            latest_metric_seq = max((m.seq for m in step_metrics), default=0)
            metric_categories = categories_for({m.metric_name for m in step_metrics})
            annotations = annotations_for_step(
                [e.to_record() for e in getattr(self, "_pipeline_events", [])],
                step_name, rec.start_time,
            )
            metrics = [
                {
                    "seq": m.seq,
                    "name": m.metric_name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "global_step": m.global_step,
                }
                for m in step_metrics
                if m.seq > since_seq
            ]
            return {
                "name": rec.name,
                "status": rec.status.value,
                "start_time": rec.start_time,
                "end_time": rec.end_time,
                "duration": (rec.end_time - rec.start_time) if rec.start_time and rec.end_time else None,
                "target_metric": rec.target_metric,
                "metric_kind": rec.metric_kind,
                "verdict": rec.verdict,
                "metrics": metrics,
                "metric_categories": metric_categories,
                "annotations": annotations,
                "latest_metric_seq": latest_metric_seq,
                "snapshot": rec.snapshot,
                "snapshot_key_kinds": rec.snapshot_key_kinds,
                "snapshot_etag": build_snapshot_etag(rec, latest_metric_seq),
                "error": rec.error,
            }

    def get_step_snapshot_etag(self, step_name: str) -> str | None:
        with self._lock:
            rec = self._steps.get(step_name)
            if rec is None:
                return None
            latest_metric_seq = max(
                (m.seq for m in self._metrics if m.step_name == step_name),
                default=0,
            )
            return build_snapshot_etag(rec, latest_metric_seq)

    def _broadcast_pipeline_overview(self) -> None:
        overview = None
        built = False
        with best_effort("build pipeline overview for broadcast", logger=logger):
            overview = self.get_pipeline_overview()
            built = True
        if not built:
            return
        self._broadcast({"type": "pipeline_overview", **overview})
