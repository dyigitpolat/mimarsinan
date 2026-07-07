"""Step timeline (Gantt) with endpoint step-budget shading and simulator split."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

_SIMULATOR_GROUP = "simulation"


def build_gantt(
    steps: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]] = (),
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Rows of per-step wall bars + the endpoint step-budget ledger summary.

    The simulator flag mirrors the tier acceptance arithmetic: artifact wall
    excludes simulation steps (the "artifact vs total wall" toggle).
    """
    profile_wall = {
        (record.get("payload") or {}).get("step"): (record.get("payload") or {}).get("wall_s")
        for record in events
        if record.get("kind") == "profile"
    }
    rows: List[Dict[str, Any]] = []
    origin = None
    for step in steps:
        start = step.get("start_time")
        if start is not None and (origin is None or start < origin):
            origin = start
    for step in steps:
        start = step.get("start_time")
        wall = step.get("duration")
        if wall is None:
            wall = profile_wall.get(step.get("name"))
        if start is None or wall is None:
            continue
        rows.append({
            "step": step.get("name"),
            "offset_s": float(start) - float(origin if origin is not None else start),
            "wall_s": float(wall),
            "simulator": step.get("semantic_group") == _SIMULATOR_GROUP,
            "group": step.get("semantic_group") or "other",
            "status": step.get("status"),
        })

    artifact_wall = sum(r["wall_s"] for r in rows if not r["simulator"])
    total_wall = sum(r["wall_s"] for r in rows)

    endpoint = _endpoint_budget(events, config or {})
    return {
        "rows": rows,
        "artifact_wall_s": artifact_wall,
        "total_wall_s": total_wall,
        "endpoint_budget": endpoint,
    }


def _endpoint_budget(
    events: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """The run-total endpoint step ledger vs its budget, per engaged stage."""
    stages: List[Dict[str, Any]] = []
    consumed = 0
    for record in events:
        if record.get("kind") != "mbh_endpoint":
            continue
        payload = record.get("payload") or {}
        steps_used = int(payload.get("steps_used") or 0)
        consumed += steps_used
        stages.append({
            "step": record.get("step"),
            "tuner": payload.get("tuner"),
            "steps_used": steps_used,
            "budget_steps": payload.get("budget_steps"),
            "engaged": payload.get("engaged"),
            "reached": payload.get("reached"),
        })
    budget = config.get("endpoint_floor_steps")
    return {
        "stages": stages,
        "consumed_steps": consumed,
        "budget_steps": int(budget) if budget is not None else None,
    }
