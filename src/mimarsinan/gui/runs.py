"""Discovery and loading of previous pipeline runs from the generated files directory."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from mimarsinan.gui.persistence import load_persisted_steps, load_live_metrics

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def get_runs_root() -> str:
    return os.environ.get("MIMARSINAN_RUNS_ROOT", "./generated")


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
    steps = []
    for name, sd in steps_data.items():
        start_t = sd.get("start_time")
        end_t = sd.get("end_time")
        steps.append({
            "name": name,
            "status": "completed" if end_t else "pending",
            "start_time": start_t,
            "end_time": end_t,
            "duration": (end_t - start_t) if start_t and end_t else None,
            "target_metric": sd.get("target_metric"),
        })
    return {
        "steps": steps,
        "current_step": None,
        "config": config,
    }


def get_run_step_detail(run_id: str, step_name: str) -> dict[str, Any] | None:
    """Load detailed step data (metrics, snapshot) from a past run."""
    _validate_run_id(run_id)
    run_dir = Path(get_runs_root()) / run_id
    steps_data = load_persisted_steps(str(run_dir))
    sd = steps_data.get(step_name)
    if sd is None:
        return None
    return {
        "name": step_name,
        "status": "completed" if sd.get("end_time") else "pending",
        "start_time": sd.get("start_time"),
        "end_time": sd.get("end_time"),
        "duration": (sd.get("end_time", 0) - sd.get("start_time", 0))
            if sd.get("start_time") and sd.get("end_time") else None,
        "target_metric": sd.get("target_metric"),
        "metrics": sd.get("metrics", []),
        "snapshot": sd.get("snapshot"),
        "snapshot_key_kinds": sd.get("snapshot_key_kinds"),
    }
