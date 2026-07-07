"""Write persisted GUI state to disk."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from mimarsinan.gui.runtime.persistence.paths import (
    atomic_write_json,
    console_log_path,
    live_metrics_path,
    run_info_path,
    steps_file_lock,
    steps_path,
)
from mimarsinan.gui.runtime.persistence.resource_paths import resource_disk_path

logger = logging.getLogger("mimarsinan.gui")


def write_persisted_steps_replace(working_directory: str, steps: dict[str, Any]) -> None:
    path = steps_path(working_directory)
    with steps_file_lock(path):
        atomic_write_json(path, {"steps": steps})


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
    path = steps_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with steps_file_lock(path):
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
        atomic_write_json(path, {"steps": existing})


def save_step_status(
    working_directory: str,
    step_name: str,
    *,
    status: str,
    end_time: float | None = None,
    target_metric: float | None = None,
) -> None:
    path = steps_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with steps_file_lock(path):
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                existing = data.get("steps", {})
            except (OSError, json.JSONDecodeError):
                pass

        entry = dict(existing.get(step_name) or {})
        entry["status"] = status
        if end_time is not None:
            entry["end_time"] = end_time
        if target_metric is not None:
            entry["target_metric"] = target_metric
        existing[step_name] = entry
        atomic_write_json(path, {"steps": existing})


def save_resource_to_disk(
    working_directory: str,
    step_name: str,
    kind: str,
    rid: str,
    payload: bytes,
    *,
    media_type: str,
) -> None:
    path = resource_disk_path(working_directory, step_name, kind, rid, media_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
        tmp.replace(path)
    except OSError as e:
        logger.debug("Failed to write resource %s: %s", path, e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def save_run_info(
    working_directory: str,
    pid: int,
    step_names: list[str],
    config_summary: dict[str, Any] | None = None,
) -> None:
    path = run_info_path(working_directory)
    info = {
        "pid": pid,
        "step_names": step_names,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "config_summary": config_summary or {},
        "error": None,
    }
    atomic_write_json(path, info)


def update_run_status(
    working_directory: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    path = run_info_path(working_directory)
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
    atomic_write_json(path, info)


def append_live_metric(
    working_directory: str,
    step_name: str,
    metric_name: str,
    value: float,
    seq: int,
    timestamp: float,
) -> None:
    append_live_metrics(working_directory, [{
        "step": step_name,
        "name": metric_name,
        "value": value,
        "seq": seq,
        "timestamp": timestamp,
    }])


def append_live_metrics(working_directory: str, records: list) -> None:
    """Append a BATCH of metric records with one open/write (the trainer
    reports every optimizer step; per-record appends on a network filesystem
    cost more wall than the training itself)."""
    if not records:
        return
    path = live_metrics_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record) + "\n" for record in records)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload)
    except OSError as e:
        logger.debug("Failed to append %d metrics to %s: %s", len(records), path, e)


def append_console_log(
    working_directory: str,
    stream: str,
    line: str,
    ts: float,
) -> None:
    path = console_log_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"stream": stream, "line": line, "ts": ts}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.debug("Failed to append console log to %s: %s", path, e)
