"""Deterministic SANA-FE floorplan resolution from the DECLARED platform.

The floorplan (cores per tile + tile grid) is a pure function of the declared
platform, never of the packed model: two runs of different models on the same
declared platform produce identical floorplans, so cross-run NoC metrics are
comparable by construction.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, NamedTuple

# Physical tile wiring of the bundled SANA-FE reference architectures:
# sana_fe/arch/loihi.yaml packs ``loihi_core[0..3]`` per tile (4 cores/tile,
# 8x4 tiles); sana_fe/arch/truenorth.yaml defines one ``truenorth_core`` per
# tile (1 core/tile, 64x64 tiles). Presets absent here (e.g. ``custom``)
# carry no floorplan fact and fall back to ceil(sqrt(declared)).
PRESET_CORES_PER_TILE: Mapping[str, int] = {
    "loihi": 4,
    "truenorth": 1,
}


class Floorplan(NamedTuple):
    """Resolved NoC floorplan: ``rows*cols`` tiles of ``cores_per_tile`` cores."""

    cores_per_tile: int
    rows: int
    cols: int


def _mesh_dims(n_tiles: int) -> tuple[int, int]:
    """Most-square exact factorization ``(width>=height, width*height==n_tiles)``.

    Must be exact: a ceil-padded mesh leaves phantom tiles the YAML never defines
    and SANA-FE's C++ NoC then SIGFPEs indexing them.
    """
    n = max(1, int(n_tiles))
    height = 1
    for h in range(int(math.isqrt(n)), 0, -1):
        if n % h == 0:
            height = h
            break
    return n // height, height


def floorplan_config_errors(constraints: Mapping[str, Any]) -> List[str]:
    """Cross-key floorplan declaration errors (the standard validation rule).

    A half-declared tile grid is a config error: ``tile_grid_rows`` and
    ``tile_grid_cols`` must be declared together (both 0 = derived).
    """
    rows = int(constraints.get("tile_grid_rows", 0) or 0)
    cols = int(constraints.get("tile_grid_cols", 0) or 0)
    if (rows > 0) != (cols > 0):
        return [
            "tile_grid_rows and tile_grid_cols must be declared together "
            f"(both 0 = derived, both > 0 = explicit); got rows={rows}, "
            f"cols={cols}"
        ]
    return []


def resolve_floorplan(
    declared_core_capacity: int,
    preset_name: str,
    cores_per_tile: int = 0,
    tile_grid_rows: int = 0,
    tile_grid_cols: int = 0,
) -> Floorplan:
    """Resolve ``(cores_per_tile, rows, cols)`` from the DECLARED platform.

    Precedence for cores per tile: an explicit ``cores_per_tile`` wins; else
    the preset's physical tile wiring (``PRESET_CORES_PER_TILE`` — loihi.yaml
    packs 4 cores per tile, truenorth.yaml packs 1); else
    ``ceil(sqrt(declared_core_capacity))``. An explicit ``tile_grid_rows`` x
    ``tile_grid_cols`` grid wins over the derived most-square exact grid of
    ``n_tiles = ceil(declared / cores_per_tile)`` tiles.

    Hard invariant (loud ValueError): an explicit grid is valid iff it holds
    the declared capacity (``rows*cols*cores_per_tile >=
    declared_core_capacity``). An OVERSIZED explicit grid is legitimate — a
    fixed physical chip whose surplus tiles sit idle — and safe: the arch
    builder defines every ``rows*cols`` tile, so no tile the NoC can walk is
    phantom (phantom tiles SIGFPE SANA-FE's C++ NoC). The derived grid needs
    no runtime check: it is an exact factorization holding the capacity by
    construction.
    """
    declared = int(declared_core_capacity)
    if declared <= 0:
        raise ValueError(
            "declared_core_capacity must be >= 1 to resolve a floorplan; "
            f"got {declared}"
        )
    cpt = int(cores_per_tile or 0)
    rows = int(tile_grid_rows or 0)
    cols = int(tile_grid_cols or 0)
    if cpt < 0 or rows < 0 or cols < 0:
        raise ValueError(
            "floorplan keys must be >= 0 (0 = derived); got "
            f"cores_per_tile={cpt}, tile_grid_rows={rows}, tile_grid_cols={cols}"
        )
    errors = floorplan_config_errors(
        {"tile_grid_rows": rows, "tile_grid_cols": cols}
    )
    if errors:
        raise ValueError(errors[0])

    if cpt == 0:
        cpt = int(PRESET_CORES_PER_TILE.get(str(preset_name), 0))
    if cpt == 0:
        cpt = max(1, math.isqrt(declared))
        if cpt * cpt < declared:
            cpt += 1

    if rows > 0:
        # Explicit grid: a FIXED physical chip. The one REAL validation is
        # capacity — oversized (idle tiles) is legitimate, and safe because
        # every rows*cols tile is defined by the arch builder (the SIGFPE
        # invariant is pinned at the YAML level in test_sanafe_arch_synth).
        if rows * cols * cpt < declared:
            raise ValueError(
                f"floorplan capacity {rows}x{cols} tiles x {cpt} cores/tile = "
                f"{rows * cols * cpt} cores cannot hold the declared platform "
                f"capacity of {declared} cores"
            )
    else:
        # Derived grid: exact most-square factorization of ceil(declared/cpt)
        # tiles — exactness (no phantom tiles) and capacity both hold by
        # construction, so nothing runtime-checkable remains on this path.
        n_tiles = -(-declared // cpt)
        width, height = _mesh_dims(n_tiles)
        cols, rows = width, height
    return Floorplan(cores_per_tile=cpt, rows=rows, cols=cols)
