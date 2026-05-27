"""Path helpers and atomic I/O for GUI state on disk."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

GUI_STATE_DIR = "_GUI_STATE"
STEPS_FILENAME = "steps.json"
RUN_INFO_FILENAME = "run_info.json"
LIVE_METRICS_FILENAME = "live_metrics.jsonl"
CONSOLE_LOG_FILENAME = "console.jsonl"
RESOURCES_DIRNAME = "resources"

_STEPS_FILE_LOCKS_LOCK = threading.Lock()
_STEPS_FILE_LOCKS: dict[str, threading.Lock] = {}


def gui_state_dir(working_directory: str) -> Path:
    return Path(working_directory) / GUI_STATE_DIR


def steps_path(working_directory: str) -> Path:
    return gui_state_dir(working_directory) / STEPS_FILENAME


def run_info_path(working_directory: str) -> Path:
    return gui_state_dir(working_directory) / RUN_INFO_FILENAME


def live_metrics_path(working_directory: str) -> Path:
    return gui_state_dir(working_directory) / LIVE_METRICS_FILENAME


def console_log_path(working_directory: str) -> Path:
    return gui_state_dir(working_directory) / CONSOLE_LOG_FILENAME


def steps_file_lock(path: Path) -> threading.Lock:
    key = str(path.resolve() if path.exists() else path.absolute())
    with _STEPS_FILE_LOCKS_LOCK:
        lock = _STEPS_FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _STEPS_FILE_LOCKS[key] = lock
        return lock


def atomic_write_json(path: Path, data: Any) -> None:
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
