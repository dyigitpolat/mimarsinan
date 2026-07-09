"""Display/serialization helpers: safe numeric conversion and layer grouping keys."""

from __future__ import annotations

import re
from typing import Any


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert value to float for display/serialization; return default when unconvertible."""
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return default


_RE_CONV_POS = re.compile(r"^(.*)_pos\d+_\d+_g\d+$")
_RE_FC_TILE = re.compile(r"^(.*)_tile_\d+_\d+$")


def layer_key_from_node_name(name: str) -> str:
    """Best-effort grouping key that collapses per-position/per-tile cores into a layer stack."""
    s = str(name)
    m = _RE_CONV_POS.match(s)
    if m:
        return m.group(1)
    m = _RE_FC_TILE.match(s)
    if m:
        return m.group(1)
    if "_psum_" in s:
        return s.split("_psum_", 1)[0]
    return s
