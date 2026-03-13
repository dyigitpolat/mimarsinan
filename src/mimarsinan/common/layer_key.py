"""Display/serialization helper: layer grouping key from IR or mapper node names."""

from __future__ import annotations

import re

_RE_CONV_POS = re.compile(r"^(.*)_pos\d+_\d+_g\d+$")
_RE_FC_TILE = re.compile(r"^(.*)_tile_\d+_\d+$")


def layer_key_from_node_name(name: str) -> str:
    """
    Best-effort grouping key that collapses per-position/per-tile cores into a layer stack.

    Used by GUI snapshot and visualization (mapping_graphviz) for consistent
    layer naming in topology views.
    """
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
