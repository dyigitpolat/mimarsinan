"""Run-directory adaptation instrumentation (W3-S1): artifacts only.

Two NEW files under the pipeline working directory, written at tuner-step
commit time and read by nothing in the training path:

- ``ft_pass_walls.json`` — the AC5 per-fine-tuning-PASS wall bundle,
  accumulated ACROSS steps under step-qualified labels
  (``"<StepName>/<label>"``). The shape is pinned to the pre-existing reader
  ``chip_simulation.cost_extraction._ft_pass_walls_from_run``
  (``{"max_ft_pass_wall_s", "passes"}``), whose filename constant is the SSOT.
- ``retention_ledger.json`` — one entry per tuner-hosting step: the step's
  entry/exit metrics, the retention envelope, and the endpoint-step ledger
  accounting the W3 reallocation program needs.

Both writers are read-modify-write with an atomic replace, so a concurrent
reader never observes a torn file. Under [MBH-DRAWS] best-of-N the persisted
walls are the KEPT draw's — the trajectory the committed artifact experienced.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mimarsinan.chip_simulation.cost_extraction import FT_PASS_WALLS_FILENAME
from mimarsinan.tuning.orchestration import endpoint_steps, retention_envelope
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY

RETENTION_LEDGER_FILENAME = "retention_ledger.json"


def _read_json_or(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_write_json(path: str, payload: Any) -> None:
    """Serialize fully into a sibling temp file, then land it in one replace."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def merge_ft_pass_walls(
    working_directory: str, step_name: str, metrics: dict
) -> dict | None:
    """Merge one step's tuner ``ft_pass_wall_metrics()`` into the run's
    ``ft_pass_walls.json`` (read-modify-write, atomic replace).

    Passes accumulate across steps as ``"<StepName>/<label>"``;
    ``max_ft_pass_wall_s`` is recomputed over ALL accumulated passes. A tuner
    that ran zero FT passes leaves the run byte-identical (no file is created
    or touched); returns the merged payload, or ``None`` when nothing was
    written.
    """
    new_passes = [
        {"label": f"{step_name}/{p['label']}", "wall_s": float(p["wall_s"])}
        for p in (metrics.get("passes") or [])
    ]
    if not new_passes:
        return None
    path = os.path.join(working_directory, FT_PASS_WALLS_FILENAME)
    existing = _read_json_or(path, {}) or {}
    passes = list(existing.get("passes") or []) + new_passes
    payload = {
        "max_ft_pass_wall_s": max(float(p["wall_s"]) for p in passes),
        "passes": passes,
    }
    _atomic_write_json(path, payload)
    return payload


def resolve_endpoint_steps_total(pipeline) -> int:
    """The run's endpoint-step budget, resolved exactly as the armed endpoint
    stage resolves it (config key over the ``TUNING_POLICY`` default)."""
    return int(
        pipeline.config.get("endpoint_floor_steps", TUNING_POLICY.endpoint_floor_steps)
    )


def retention_entry(
    *,
    step_name: str,
    entry_metric: float | None,
    exit_metric: float,
    pipeline,
    consumed_before: int | None,
) -> dict:
    """One tuner-hosting step's retention accounting.

    ``armed_recovery`` derives from the endpoint-step ledger delta: the ledger
    consume in ``frontier/endpoint_recovery`` happens ONLY on an armed stage,
    so a positive delta is the arming signature (an armed stage that trained
    zero steps is indistinguishable from a non-armed one — by design the cheap
    gauge, never a reach into tuner internals).
    """
    consumed_after = endpoint_steps.consumed(pipeline)
    entry = None if entry_metric is None else float(entry_metric)
    return {
        "step": str(step_name),
        "entry_metric": entry,
        "exit_metric": float(exit_metric),
        "retention_delta": None if entry is None else float(exit_metric) - entry,
        "envelope": retention_envelope.peek(pipeline),
        "endpoint_steps_consumed_before": (
            None if consumed_before is None else int(consumed_before)
        ),
        "endpoint_steps_consumed_after": int(consumed_after),
        "endpoint_steps_total": resolve_endpoint_steps_total(pipeline),
        "armed_recovery": bool(
            consumed_before is not None and consumed_after > int(consumed_before)
        ),
    }


def append_retention_entry(working_directory: str, entry: dict) -> dict:
    """Append one step's entry to ``retention_ledger.json`` (read-modify-write,
    atomic replace); entries stay in step order. Returns the written payload."""
    path = os.path.join(working_directory, RETENTION_LEDGER_FILENAME)
    existing = _read_json_or(path, {}) or {}
    payload = {"entries": list(existing.get("entries") or []) + [entry]}
    _atomic_write_json(path, payload)
    return payload
