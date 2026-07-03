"""Shared helpers for snapshot builders: numeric/dict conversion and cache key mapping."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mimarsinan.common.best_effort import best_effort

logger = logging.getLogger("mimarsinan.gui")


def _t(val: Any) -> float:
    """Convert tensor/ndarray/scalar to plain float."""
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def _histogram(arr: np.ndarray, bins: int = 50) -> dict:
    """Return a compact histogram dict (counts, bin_edges)."""
    if arr.size == 0:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(arr.flatten(), bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


def _safe_scalar(obj: Any, attr: str) -> float | None:
    result: float | None = None
    with best_effort(f"read scalar attribute {attr!r}", logger=logger):
        val = getattr(obj, attr, None)
        if val is not None:
            result = _t(val)
    return result


def _safe_dict(obj: Any) -> Any:
    """Recursively convert an object to JSON-safe types."""
    from mimarsinan.gui.json_util import to_json_safe

    return to_json_safe(obj)


# Map cache virtual key (step contract) to snapshot key (GUI tab).
_CACHE_KEY_TO_SNAPSHOT_KEY: dict[str, str] = {
    "model": "model",
    "fused_model": "model",
    "ir_graph": "ir_graph",
    "hard_core_mapping": "hard_core_mapping",
    "architecture_search_result": "search_result",
    "adaptation_manager": "adaptation_manager",
    "activation_scales": "activation_scales",
    "platform_constraints_resolved": "platform_constraints",
    "sanafe_simulation_results": "sanafe_simulation",
}
