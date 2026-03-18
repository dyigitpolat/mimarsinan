"""Persistence of GUI state for backfill and process-based monitoring.

Files written under ``<working_dir>/_GUI_STATE/``:

- ``steps.json``          – per-step snapshot (metrics, snapshot, status)
- ``run_info.json``       – run metadata (pid, step_names, status, config_summary)
- ``live_metrics.jsonl``  – append-only stream of metric events for live monitoring
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

_GUI_STATE_DIR = "_GUI_STATE"
_STEPS_FILENAME = "steps.json"
_RUN_INFO_FILENAME = "run_info.json"
_LIVE_METRICS_FILENAME = "live_metrics.jsonl"


def _gui_state_dir(working_directory: str) -> Path:
    return Path(working_directory) / _GUI_STATE_DIR


def _state_path(working_directory: str) -> Path:
    return _gui_state_dir(working_directory) / _STEPS_FILENAME


def _run_info_path(working_directory: str) -> Path:
    return _gui_state_dir(working_directory) / _RUN_INFO_FILENAME


def _live_metrics_path(working_directory: str) -> Path:
    return _gui_state_dir(working_directory) / _LIVE_METRICS_FILENAME


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except OSError as e:
        logger.debug("Failed to write %s: %s", path, e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ── steps.json ────────────────────────────────────────────────────────────

def load_persisted_steps(working_directory: str) -> dict[str, Any]:
    """Load persisted step data from working_directory/_GUI_STATE/steps.json.

    Returns a dict keyed by step name; each value has start_time, end_time,
    target_metric, metrics, snapshot, snapshot_key_kinds. Returns {} if file
    is missing or invalid.
    """
    path = _state_path(working_directory)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("steps", {})
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to load persisted GUI steps from %s: %s", path, e)
        return {}


def save_step_to_persisted(
    working_directory: str,
    step_name: str,
    start_time: float | None,
    end_time: float | None,
    target_metric: float | None,
    metrics: list[dict],
    snapshot: dict | None,
    snapshot_key_kinds: dict | None,
    *,
    status: str | None = None,
) -> None:
    """Merge one step's data into persisted steps.json and write atomically."""
    path = _state_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("steps", {})
        except (OSError, json.JSONDecodeError):
            pass

    entry: dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
        "target_metric": target_metric,
        "metrics": metrics,
        "snapshot": snapshot,
        "snapshot_key_kinds": snapshot_key_kinds or {},
    }
    if status is not None:
        entry["status"] = status

    existing[step_name] = entry
    _atomic_write_json(path, {"steps": existing})


# ── run_info.json ─────────────────────────────────────────────────────────

def save_run_info(
    working_directory: str,
    pid: int,
    step_names: list[str],
    config_summary: dict[str, Any] | None = None,
) -> None:
    """Write initial run_info.json for a headless process."""
    path = _run_info_path(working_directory)
    info = {
        "pid": pid,
        "step_names": step_names,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "config_summary": config_summary or {},
        "error": None,
    }
    _atomic_write_json(path, info)


def update_run_status(
    working_directory: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """Update the status field in run_info.json."""
    path = _run_info_path(working_directory)
    info: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    info["status"] = status
    info["finished_at"] = time.time()
    if error is not None:
        info["error"] = error
    _atomic_write_json(path, info)


def load_run_info(working_directory: str) -> dict[str, Any] | None:
    """Load run_info.json. Returns None if missing."""
    path = _run_info_path(working_directory)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── live_metrics.jsonl ────────────────────────────────────────────────────

def append_live_metric(
    working_directory: str,
    step_name: str,
    metric_name: str,
    value: float,
    seq: int,
    timestamp: float,
) -> None:
    """Append a single metric event to live_metrics.jsonl."""
    path = _live_metrics_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step_name,
        "name": metric_name,
        "value": value,
        "seq": seq,
        "timestamp": timestamp,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.debug("Failed to append metric to %s: %s", path, e)


def load_live_metrics(
    working_directory: str,
    *,
    step_name: str | None = None,
) -> list[dict[str, Any]]:
    """Read live_metrics.jsonl, optionally filtering by step."""
    path = _live_metrics_path(working_directory)
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if step_name is not None and record.get("step") != step_name:
                    continue
                results.append(record)
    except OSError:
        pass
    return results
