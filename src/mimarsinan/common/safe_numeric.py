"""Display/serialization helpers: safe conversion to float for plots and labels."""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert value to float for display/serialization; return default when unconvertible."""
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return default
