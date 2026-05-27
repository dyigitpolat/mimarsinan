"""GUI snapshot module."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.util.helpers import _t, _histogram, _safe_scalar, _safe_dict, _CACHE_KEY_TO_SNAPSHOT_KEY
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.gui.resources import ResourceDescriptor

# Bump cautiously: frontend URL builders hard-code these.
RESOURCE_KIND_IR_CORE_HEATMAP = "ir_core_heatmap"
RESOURCE_KIND_IR_CORE_PRE_PRUNING = "ir_core_pre_pruning"
RESOURCE_KIND_IR_CORE_BIAS = "ir_core_bias"
RESOURCE_KIND_IR_BANK_HEATMAP = "ir_bank_heatmap"
RESOURCE_KIND_HARD_CORE_HEATMAP = "hard_core_heatmap"
RESOURCE_KIND_CONNECTIVITY = "connectivity"
RESOURCE_KIND_PRUNING_LAYER_HEATMAP = "pruning_layer_heatmap"


# Per-NeuralCore liveness tags surfaced in the GUI (must match
# ``mimarsinan.mapping.pruning.ir_liveness.NodeLiveness`` for current runs).
LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"  # only for old pickles still containing (1,1) placeholders

def snapshot_search_result(result: Any) -> dict:
    """Extract Pareto front, candidates, and objectives from a SearchResult."""
    if isinstance(result, dict):
        return _snapshot_search_result_dict(result)
    return _snapshot_search_result_obj(result)


def _snapshot_search_result_dict(d: dict) -> dict:
    """Handle the dict-serialized SearchResult form."""
    best = None
    try:
        b = d["best"]
        best = {
            "config": _safe_dict(b.get("configuration", b.get("config", {}))),
            "objectives": _safe_dict(b.get("objectives", {})),
        }
    except Exception:
        logger.debug("Failed to extract best from dict search result", exc_info=True)

    pareto = []
    try:
        for c in d.get("pareto_front", []):
            pareto.append({
                "config": _safe_dict(c.get("configuration", c.get("config", {}))),
                "objectives": _safe_dict(c.get("objectives", {})),
            })
    except Exception:
        logger.debug("Failed to extract pareto_front from dict search result", exc_info=True)

    history = []
    try:
        for h in d.get("history", []):
            history.append(_safe_dict(h))
    except Exception:
        logger.debug("Failed to extract history from dict search result", exc_info=True)

    objectives = []
    try:
        for obj in d.get("objectives", []):
            if isinstance(obj, dict):
                objectives.append({"name": obj.get("name", "?"), "goal": obj.get("goal", "?")})
            else:
                objectives.append({"name": getattr(obj, "name", "?"), "goal": getattr(obj, "goal", "?")})
    except Exception:
        logger.debug("Failed to extract objectives from dict search result", exc_info=True)

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
    try:
        best = {
            "config": _safe_dict(result.best.config if hasattr(result.best, 'config') else result.best.configuration),
            "objectives": _safe_dict(result.best.objectives),
        }
    except Exception:
        best = None

    pareto = []
    try:
        for c in result.pareto_front:
            pareto.append({
                "config": _safe_dict(c.config if hasattr(c, 'config') else c.configuration),
                "objectives": _safe_dict(c.objectives),
            })
    except Exception:
        pass

    history = []
    try:
        for h in result.history:
            history.append(_safe_dict(h))
    except Exception:
        pass

    objectives = []
    try:
        for obj in result.objectives:
            objectives.append({"name": obj.name, "goal": obj.goal})
    except Exception:
        pass

    return {
        "best": best,
        "pareto_front": pareto,
        "num_candidates": len(getattr(result, "all_candidates", [])),
        "history": history,
        "objectives": objectives,
    }

