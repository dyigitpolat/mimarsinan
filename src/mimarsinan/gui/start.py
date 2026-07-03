"""Start the GUI server and optional step backfill."""

from __future__ import annotations

from typing import Any

from mimarsinan.gui.handle import GUIHandle
from mimarsinan.gui.resources import ResourceStore
from mimarsinan.gui.runtime.collector import DataCollector, to_json_safe
from mimarsinan.gui.runtime.persistence import (
    load_persisted_steps,
    write_persisted_steps_replace,
)
from mimarsinan.gui.server import start_server
from mimarsinan.gui.snapshot import build_step_snapshot


def start_gui(
    pipeline: Any,
    *,
    port: int = 8501,
    host: str = "0.0.0.0",
    start_step: str | None = None,
) -> GUIHandle:
    collector = DataCollector()
    collector.set_resource_store(ResourceStore())

    step_names = [name for name, _ in pipeline.steps]
    config = getattr(pipeline, "config", {})
    safe_config = to_json_safe(config)
    collector.set_pipeline_info(step_names, safe_config)

    if start_step is not None:
        backfill_skipped_steps(pipeline, collector, step_names, start_step)

    collector.set_working_directory(getattr(pipeline, "working_directory", None))

    start_server(collector, host=host, port=port)
    return GUIHandle(pipeline, collector)


def backfill_skipped_steps(
    pipeline: Any,
    collector: DataCollector,
    step_names: list[str],
    start_step: str,
) -> None:
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
                snapshot, snapshot_key_kinds, resource_descriptors = build_step_snapshot(
                    pipeline, step_name, step=step
                )
            except Exception:
                snapshot = None
                snapshot_key_kinds = None
                resource_descriptors = []
            collector.step_completed(
                step_name,
                target_metric=None,
                snapshot=snapshot,
                snapshot_key_kinds=snapshot_key_kinds,
                resources=resource_descriptors,
            )

    if working_dir:
        _persist_skipped_steps_to_steps_json(working_dir, collector, step_names, start_idx)


def _persist_skipped_steps_to_steps_json(
    working_dir: str,
    collector: DataCollector,
    step_names: list[str],
    start_idx: int,
) -> None:
    merged: dict[str, Any] = {}
    for i in range(start_idx):
        name = step_names[i]
        detail = collector.get_step_detail(name)
        if not detail:
            continue
        merged[name] = {
            "start_time": detail.get("start_time"),
            "end_time": detail.get("end_time"),
            "target_metric": detail.get("target_metric"),
            "metrics": detail.get("metrics", []),
            "snapshot": detail.get("snapshot"),
            "snapshot_key_kinds": detail.get("snapshot_key_kinds") or {},
            "status": "completed",
        }
    write_persisted_steps_replace(working_dir, merged)
