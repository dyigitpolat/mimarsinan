"""Shared helpers for snapshot builders: numeric/dict conversion and cache key mapping."""

from __future__ import annotations

from typing import Any

import numpy as np


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
    try:
        val = getattr(obj, attr, None)
        if val is None:
            return None
        return _t(val)
    except Exception:
        return None


def _safe_dict(obj: Any) -> Any:
    """Recursively convert an object to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dict(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


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
}
