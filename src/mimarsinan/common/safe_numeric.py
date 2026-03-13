"""Display/serialization helpers: safe conversion to float for plots and labels."""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Convert value to float for display/serialization; return default on failure.

    Used by visualization (search_visualization, mapping_graphviz) and any
    code that needs to coerce tensors/scalars to float for JSON or labels.
    """
    try:
        return float(value)
    except Exception:
        return default
