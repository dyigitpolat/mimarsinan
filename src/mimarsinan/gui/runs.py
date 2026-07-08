"""Discovery and loading of previous pipeline runs from the generated files directory."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.env import runs_root
from mimarsinan.gui.runtime.persistence import (
    load_console_logs,
    load_events,
    load_persisted_steps,
)
from mimarsinan.gui.snapshot.console_events import parse_console_events
from mimarsinan.gui.viewmodel import (
    annotations_for_step,
    build_overview_chart,
    categories_for,
    persisted_step_view,
    semantic_groups_from_config_view,
    step_bar_badge,
)
from mimarsinan.gui.snapshot.rebuild import rebuild_step_snapshot_from_disk

logger = logging.getLogger("mimarsinan.gui")

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def suggest_resume_step(ordered_steps: list[str], completed_steps: set[str]) -> str | None:
    """Return the first step in canonical order that is not yet completed, or None if all are complete."""
    for step in ordered_steps:
        if step not in completed_steps:
            return step
    return None


def get_runs_root() -> str:
    return runs_root()


def _validate_run_id(run_id: str) -> str:
    if not _SAFE_ID_RE.match(run_id):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return run_id


def list_runs(*, include_steps: bool = False) -> list[dict[str, Any]]:
    """Scan the generated files root for past pipeline runs.

    A valid run is a directory containing ``_RUN_CONFIG/config.json``.
    """
    root = Path(get_runs_root())
    if not root.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        config_path = child / "_RUN_CONFIG" / "config.json"
        if not config_path.exists():
            continue
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        entry: dict[str, Any] = {
            "run_id": child.name,
            "experiment_name": config.get("experiment_name", child.name),
            "pipeline_mode": config.get("pipeline_mode", "unknown"),
            "created_at": child.stat().st_mtime,
        }
        if include_steps:
            steps_data = load_persisted_steps(str(child))
            entry["steps"] = list(steps_data.keys())
            entry["total_steps"] = len(entry["steps"])
            entry["completed_steps"] = sum(
                1 for s in steps_data.values()
                if s.get("end_time") is not None
            )
        results.append(entry)
    return results


def get_run_config(run_id: str) -> dict[str, Any] | None:
    """Load the full deployment config for a past run."""
    _validate_run_id(run_id)
    config_path = Path(get_runs_root()) / run_id / "_RUN_CONFIG" / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def get_run_pipeline(run_id: str) -> dict[str, Any] | None:
    """Load pipeline overview from a past run's persisted state."""
    _validate_run_id(run_id)
    run_dir = Path(get_runs_root()) / run_id
    if not run_dir.is_dir():
        return None
    steps_data = load_persisted_steps(str(run_dir))
    if not steps_data:
        return None
    config = get_run_config(run_id) or {}
    config_view = _build_run_config_view(config)
    groups = semantic_groups_from_config_view(config_view)
    steps = []
    for name, sd in steps_data.items():
        steps.append({
            **persisted_step_view(
                name, sd, status="completed" if sd.get("end_time") else "pending",
            ),
            "semantic_group": groups.get(name),
        })
    for step in steps:
        step["badge"] = step_bar_badge(step)
    return {
        "steps": steps,
        "current_step": None,
        "config": config,
        "config_view": config_view,
        "overview_chart": build_overview_chart(steps),
    }


def _build_run_config_view(config: dict) -> dict | None:
    if not config:
        return None
    result = None
    with best_effort("build run config display view", logger=logger):
        from mimarsinan.config_schema.display_view import build_config_display_view
        result = build_config_display_view(config, saved_config=config)
    return result


def get_run_console_logs(run_id: str, offset: int = 0) -> list[dict[str, Any]]:
    """Load console log entries for a past run from console.jsonl."""
    _validate_run_id(run_id)
    run_dir = Path(get_runs_root()) / run_id
    if not run_dir.is_dir():
        return []
    return load_console_logs(str(run_dir), offset=offset)


def get_run_step_detail(run_id: str, step_name: str) -> dict[str, Any] | None:
    """Load detailed step data (metrics, snapshot) from a past run."""
    _validate_run_id(run_id)
    run_dir = Path(get_runs_root()) / run_id
    steps_data = load_persisted_steps(str(run_dir))
    sd = steps_data.get(step_name)
    if sd is None:
        return None
    snapshot = sd.get("snapshot")
    snapshot_key_kinds = sd.get("snapshot_key_kinds")
    if snapshot is None:
        rebuilt = rebuild_step_snapshot_from_disk(str(run_dir), step_name)
        if rebuilt is not None:
            snapshot, snapshot_key_kinds = rebuilt
    metrics = sd.get("metrics", [])
    return {
        **persisted_step_view(
            step_name, sd, status="completed" if sd.get("end_time") else "pending",
        ),
        "metric_categories": categories_for({m.get("name", "") for m in metrics}),
        "annotations": annotations_for_step(
            get_run_events(run_id), step_name, sd.get("start_time"),
        ),
        "metrics": metrics,
        "snapshot": snapshot,
        "snapshot_key_kinds": snapshot_key_kinds,
    }


_ARTIFACT_KIND_BY_SUFFIX = {
    ".pt": "torch",
    ".pth": "torch",
    ".pickle": "pickle",
    ".pkl": "pickle",
    ".json": "json",
    ".jsonl": "jsonl",
    ".png": "image",
    ".npy": "numpy",
    ".txt": "text",
    ".log": "text",
    ".yaml": "yaml",
}

_ARTIFACT_GROUP_BY_DIR = {
    "_GUI_STATE": "monitor state",
    "_RUN_CONFIG": "config",
}


def classify_artifact(name: str) -> dict[str, Any]:
    """Kind/group/step classification for one run-directory entry name."""
    suffix = Path(name).suffix.lower()
    kind = _ARTIFACT_KIND_BY_SUFFIX.get(suffix, "other")
    group = _ARTIFACT_GROUP_BY_DIR.get(name)
    step = None
    if group is None:
        if name.startswith("segment_"):
            group = "segments"
        elif "." in name and not name.startswith((".", "_")):
            # Pipeline cache files are "<Step Name>.<cache key>.<ext>".
            head = name.split(".", 1)[0]
            if " " in head or head[:1].isupper():
                group, step = "step cache", head
            else:
                group = "run outputs"
        else:
            group = "run outputs"
    return {"kind": kind, "group": group, "step": step}


def list_dir_artifacts(run_dir: str) -> list[dict[str, Any]]:
    """Top-level run-directory inventory; directories aggregate recursively."""
    root = Path(run_dir)
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        record: dict[str, Any] = {"path": child.name, **classify_artifact(child.name)}
        try:
            if child.is_dir():
                size = 0
                count = 0
                for f in child.rglob("*"):
                    if f.is_file():
                        size += f.stat().st_size
                        count += 1
                record.update({"kind": "dir", "size": size, "files": count})
            else:
                stat = child.stat()
                record.update({"size": stat.st_size, "mtime": stat.st_mtime})
        except OSError:
            continue
        entries.append(record)
    return entries


def get_run_artifacts(run_id: str) -> list[dict[str, Any]] | None:
    """Artifact inventory for a past run, or None when the run is unknown."""
    _validate_run_id(run_id)
    run_dir = Path(get_runs_root()) / run_id
    if not run_dir.is_dir():
        return None
    return list_dir_artifacts(str(run_dir))


def get_run_artifact_file(run_id: str, rel_path: str) -> Path | None:
    """Resolve one artifact file of a past run (validated run id + safe join)."""
    _validate_run_id(run_id)
    return resolve_artifact_file(str(Path(get_runs_root()) / run_id), rel_path)


def resolve_artifact_file(run_dir: str, rel_path: str) -> Path | None:
    """Resolve one artifact file strictly inside *run_dir*; None otherwise."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        return None
    try:
        candidate = (root / rel_path).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(root):
        return None
    if not candidate.is_file():
        return None
    return candidate


def get_run_events(run_id: str, *, since_seq: int = 0) -> list[dict[str, Any]]:
    """Structured events for a past run; legacy runs backfill from console tags."""
    _validate_run_id(run_id)
    run_dir = Path(get_runs_root()) / run_id
    if not run_dir.is_dir():
        return []
    events = load_events(str(run_dir), since_seq=since_seq)
    if events:
        return events
    lines = load_console_logs(str(run_dir))
    return [e for e in parse_console_events(lines) if e.get("seq", 0) > since_seq]
