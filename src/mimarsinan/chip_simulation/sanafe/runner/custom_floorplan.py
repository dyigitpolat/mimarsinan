"""Custom-arch floorplan adoption: the loaded user arch IS the floorplan SSOT."""

from __future__ import annotations

from typing import Any, Tuple

from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import _mesh_dims


def adopt_custom_arch_floorplan(
    arch: Any,
    *,
    custom_arch_path: str,
    declared_cores_per_tile: int,
    declared_tile_grid_rows: int,
    declared_tile_grid_cols: int,
) -> Tuple[int, Tuple[int, int]]:
    """Adopt the loaded user arch's tile grouping as the floorplan.

    The div/mod core placement needs a uniform per-tile core count (a
    smaller LAST tile is the one legal remainder shape); declared floorplan
    keys that contradict the file fail loud. Returns
    ``(cores_per_tile, (mesh_width, mesh_height))`` — the arch object
    exposes no mesh, so the most-square exact grid of its tile count is
    rendered for the geometry record.
    """
    counts = [len(tile.cores) for tile in arch.tiles]
    n_tiles = len(counts)
    if n_tiles == 0:
        raise ValueError(
            f"custom arch at {custom_arch_path} defines no tiles"
        )
    cpt = int(counts[0])
    if any(int(c) != cpt for c in counts[:-1]) or int(counts[-1]) > cpt:
        raise ValueError(
            f"custom arch at {custom_arch_path} has a non-uniform "
            f"per-tile core count {counts}; the div/mod placement "
            "requires uniform tiles (a smaller last tile is allowed)"
        )
    if declared_cores_per_tile > 0 and declared_cores_per_tile != cpt:
        raise ValueError(
            f"declared cores_per_tile={declared_cores_per_tile} contradicts "
            f"the custom arch at {custom_arch_path}, which packs "
            f"{cpt} cores per tile"
        )
    declared_tiles = declared_tile_grid_rows * declared_tile_grid_cols
    if declared_tiles > 0 and declared_tiles != n_tiles:
        raise ValueError(
            f"declared tile grid {declared_tile_grid_rows}x"
            f"{declared_tile_grid_cols} = {declared_tiles} tiles contradicts "
            f"the custom arch at {custom_arch_path}, which defines "
            f"{n_tiles} tiles"
        )
    return cpt, _mesh_dims(n_tiles)
