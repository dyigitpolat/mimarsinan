"""GUI snapshot module."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.snapshot.util.helpers import _safe_dict

RESOURCE_KIND_IR_CORE_HEATMAP = "ir_core_heatmap"
RESOURCE_KIND_IR_CORE_PRE_PRUNING = "ir_core_pre_pruning"
RESOURCE_KIND_IR_CORE_BIAS = "ir_core_bias"
RESOURCE_KIND_IR_BANK_HEATMAP = "ir_bank_heatmap"
RESOURCE_KIND_HARD_CORE_HEATMAP = "hard_core_heatmap"
RESOURCE_KIND_CONNECTIVITY = "connectivity"
RESOURCE_KIND_PRUNING_LAYER_HEATMAP = "pruning_layer_heatmap"

LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"

def snapshot_search_result(result: Any) -> dict:
    """Extract Pareto front, candidates, and objectives from a SearchResult."""
    if isinstance(result, dict):
        return _snapshot_search_result_dict(result)
    return _snapshot_search_result_obj(result)


def _snapshot_search_result_dict(d: dict) -> dict:
    """Handle the dict-serialized SearchResult form."""
    best = None
    with best_effort("extract best from dict search result", logger=logger):
        b = d["best"]
        best = {
            "config": _safe_dict(b.get("configuration", b.get("config", {}))),
            "objectives": _safe_dict(b.get("objectives", {})),
        }

    pareto = []
    with best_effort("extract pareto_front from dict search result", logger=logger):
        for c in d.get("pareto_front", []):
            pareto.append({
                "config": _safe_dict(c.get("configuration", c.get("config", {}))),
                "objectives": _safe_dict(c.get("objectives", {})),
            })

    history = []
    with best_effort("extract history from dict search result", logger=logger):
        for h in d.get("history", []):
            history.append(_safe_dict(h))

    objectives = []
    with best_effort("extract objectives from dict search result", logger=logger):
        for obj in d.get("objectives", []):
            if isinstance(obj, dict):
                objectives.append({"name": obj.get("name", "?"), "goal": obj.get("goal", "?")})
            else:
                objectives.append({"name": getattr(obj, "name", "?"), "goal": getattr(obj, "goal", "?")})

    all_candidates = d.get("all_candidates", [])
    return {
        "best": best,
        "pareto_front": pareto,
        "num_candidates": len(all_candidates),
        "history": history,
        "objectives": objectives,
    }


def _snapshot_search_result_obj(result: Any) -> dict:
    """Handle the original SearchResult dataclass form."""
    best = None
    with best_effort("extract best from search result object", logger=logger):
        best = {
            "config": _safe_dict(result.best.config if hasattr(result.best, 'config') else result.best.configuration),
            "objectives": _safe_dict(result.best.objectives),
        }

    pareto = []
    with best_effort("extract pareto_front from search result object", logger=logger):
        for c in result.pareto_front:
            pareto.append({
                "config": _safe_dict(c.config if hasattr(c, 'config') else c.configuration),
                "objectives": _safe_dict(c.objectives),
            })

    history = []
    with best_effort("extract history from search result object", logger=logger):
        for h in result.history:
            history.append(_safe_dict(h))

    objectives = []
    with best_effort("extract objectives from search result object", logger=logger):
        for obj in result.objectives:
            objectives.append({"name": obj.name, "goal": obj.goal})

    return {
        "best": best,
        "pareto_front": pareto,
        "num_candidates": len(getattr(result, "all_candidates", [])),
        "history": history,
        "objectives": objectives,
    }

