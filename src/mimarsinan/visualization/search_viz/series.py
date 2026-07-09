"""Shared metric-series helpers for search-report visualizations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from mimarsinan.common.presentation import safe_float

PENALTY_CUTOFF = 1e17

__all__ = [
    "PENALTY_CUTOFF",
    "best_metric_series",
    "finite_pairs",
    "goal_by_metric",
    "nan_gapped",
    "pareto_metric_series",
]


def goal_by_metric(result_json: Dict[str, Any]) -> Dict[str, Any]:
    """Metric name -> goal from result objectives; unnamed/malformed objectives are skipped."""
    goals: Dict[str, Any] = {}
    for objective in result_json.get("objectives", []) or []:
        if not isinstance(objective, dict):
            continue
        name = objective.get("name")
        if isinstance(name, str):
            goals[name] = objective.get("goal")
    return goals


def best_metric_series(bests: List[Dict[str, Any]], name: str) -> List[float | None]:
    """Per-generation best value for ``name`` (None where missing/unconvertible)."""
    return [safe_float(b.get(name)) for b in bests]


def pareto_metric_series(pareto: List[Any], name: str) -> List[float | None]:
    """Per-candidate value for ``name`` (None for missing, unconvertible, or penalty values)."""
    vals: List[float | None] = []
    for candidate in pareto:
        objectives = (candidate.get("objectives", {}) if isinstance(candidate, dict) else {}) or {}
        value = safe_float(objectives.get(name))
        if value is None or value >= PENALTY_CUTOFF:
            vals.append(None)
        else:
            vals.append(value)
    return vals


def finite_pairs(
    a: Iterable[float | None], b: Iterable[float | None]
) -> Tuple[List[float], List[float]]:
    """Zip two optional-valued series, dropping pairs where either side is None."""
    xs: List[float] = []
    ys: List[float] = []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def nan_gapped(vals: Iterable[float | None]) -> List[float]:
    """None -> NaN so matplotlib renders the same gaps from a typed float list."""
    return [float("nan") if v is None else float(v) for v in vals]
