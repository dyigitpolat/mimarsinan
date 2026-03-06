"""Persistence of GUI step state for backfill when starting pipeline from a mid-point."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

_GUI_STATE_DIR = "_GUI_STATE"
_STEPS_FILENAME = "steps.json"


def _state_path(working_directory: str) -> Path:
    return Path(working_directory) / _GUI_STATE_DIR / _STEPS_FILENAME


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
) -> None:
    """Merge one step's data into persisted steps.json and write atomically."""
    path = _state_path(working_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("steps", {})
        except (OSError, json.JSONDecodeError):
            pass

    existing[step_name] = {
        "start_time": start_time,
        "end_time": end_time,
        "target_metric": target_metric,
        "metrics": metrics,
        "snapshot": snapshot,
        "snapshot_key_kinds": snapshot_key_kinds or {},
    }
    payload = {"steps": existing}
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(path)
    except OSError as e:
        logger.debug("Failed to write persisted GUI steps to %s: %s", path, e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
