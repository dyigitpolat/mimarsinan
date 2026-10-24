"""[E85] The cores a set of layout specs packs into, and the declared physical tiles each packed core occupies."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError

#: One packed core: its occupied cells (used axons x used neurons, the rectangle the record's ``params_bytes``
#: prices) and the physical tiles it occupies (a fused core counts its components).
PackedCore = Tuple[int, int]
#: The declared physical cores one packed core occupies, (axons, neurons) per component.
CoreTiles = Tuple[Tuple[int, int], ...]


def declared_tiles(axons: int, neurons: int, declared: Iterable[Tuple[int, int]]) -> Optional[CoreTiles]:
    """The declared physical cores one core of geometry ``axons`` x ``neurons`` occupies: itself when the geometry is
    declared, else its unique decomposition into cores of one declared type (a fused core sums the axons of cores
    of one type, ``canonical_fuse_hardcores``); None when the declared types do not determine it."""
    kinds = {(int(a), int(n)) for a, n in declared}
    if (int(axons), int(neurons)) in kinds:
        return ((int(axons), int(neurons)),)
    fits = sorted(a for a, n in kinds if n == int(neurons) and 0 < a and int(axons) % a == 0)
    return ((fits[0], int(neurons)),) * (int(axons) // fits[0]) if len(fits) == 1 else None


def core_type_tiles(axons: int, neurons: int, core_types: Sequence[LayoutHardCoreType],
                    components: Optional[Sequence[int]] = None) -> Tuple[Tuple[int, int], ...]:
    """The declared physical cores one packed core occupies: itself, or a fused core's components, read from
    ``components`` or decomposed uniquely (``declared_tiles``)."""
    if components:
        return tuple((int(a), int(neurons)) for a in components)
    declared = sorted({(int(t.max_axons), int(t.max_neurons)) for t in core_types})
    tiles = declared_tiles(axons, neurons, declared)
    if tiles is None:
        ways = sum(1 for a, n in declared if n == int(neurons) and 0 < a and int(axons) % a == 0)
        raise ResidencyPlanError(
            f"residency_plan: a fused {axons}x{neurons} core decomposes into the declared core types "
            f"{declared} in {ways} ways, so its physical tiles are not determined")
    return tiles


def tile_extents(used_axons: int, used_neurons: int, tiles: CoreTiles) -> Tuple[Tuple[int, int], ...]:
    """Per physical tile of one core, its used (rows, columns): a fused core's used rows fill its members in order
    (its axons are its members' axons concatenated), every member holding the core's used columns."""
    out, left = [], int(used_axons)
    for axons, _neurons in tiles:
        rows = min(int(axons), max(0, left))
        left -= rows
        out.append((rows, int(used_neurons)))
    return tuple(out)


def tile_rows(rows: int, tiles: CoreTiles) -> Tuple[int, ...]:
    """A count of one core's used rows (the rows its per-row sign register signs, CL-SG) per physical tile: the rows
    fill its members in order, as ``tile_extents`` fills them, and each member counts its own."""
    return tuple(r for r, _ in tile_extents(int(rows), 0, tiles))


def expand_to_tiles(cores: Sequence[Tuple[int, Optional[Tuple[int, int]], CoreTiles]], marks: Iterable[int]):
    """A pass's per-core census as one entry per physical tile: ``cores`` per core (cells, used (rows, columns) or
    None, its tiles), ``marks`` the kept core indices. Returns per tile (cells, kept tile indices, extents or None,
    tile types); a core's cells split by ``tile_extents`` where its extent is stated, else on its first tile."""
    kept = frozenset(int(m) for m in marks)
    cells: List[int] = []
    extents: List[Tuple[int, int]] = []
    types: List[Tuple[int, int]] = []
    tile_marks: List[int] = []
    for index, (core_cells, extent, tiles) in enumerate(cores):
        split = tile_extents(extent[0], extent[1], tiles) if extent is not None else None
        for j, tile in enumerate(tiles):
            if index in kept:
                tile_marks.append(len(cells))
            cells.append(split[j][0] * split[j][1] if split is not None else (int(core_cells) if j == 0 else 0))
            types.append((int(tile[0]), int(tile[1])))
            if split is not None:
                extents.append(split[j])
    stated = all(extent is not None for _, extent, _ in cores)
    return tuple(cells), frozenset(tile_marks), tuple(extents) if stated else None, tuple(types)


def pack_specs(specs: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *,
               allow_coalescing: bool, allow_splitting: bool,
               bounded: bool = False) -> Tuple[Tuple[PackedCore, ...], Tuple[CoreTiles, ...]]:
    """Every core ``specs`` pack into on the declared core types with the layout packer (``pack_layout``): per core
    its ``PackedCore`` and its physical tiles. The pool is unbounded (a layer's unit, packed alone on its best type),
    or ``bounded`` to the declared counts (a phase, packed as ``collect_noc_fragments`` packs a pass)."""
    bound = sum(
        math.ceil(s.input_count / max(1, min(t.max_axons for t in core_types)))
        * math.ceil(s.output_count / max(1, min(t.max_neurons for t in core_types))) + 1
        for s in specs
    )
    pack = pack_layout(
        softcores=list(specs),
        core_types=list(core_types) if bounded else [LayoutHardCoreType(t.max_axons, t.max_neurons, bound)
                                                     for t in core_types],
        allow_neuron_splitting=allow_splitting, allow_coalescing=allow_coalescing,
    )
    if not pack.feasible:
        raise ResidencyPlanError(
            f"residency_plan: the cores of {specs[0].name!r} pack on no declared core type ({pack.error})")
    snaps = tuple(pack.used_core_snapshots or ())
    tiles = tuple(core_type_tiles(snap.axons_per_core, snap.neurons_per_core, core_types) for snap in snaps)
    return tuple((int(s.used_axons) * int(s.used_neurons), len(t)) for s, t in zip(snaps, tiles)), tiles


def packed_cores(specs: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *,
                 allow_coalescing: bool, allow_splitting: bool, bounded: bool = False) -> Tuple[PackedCore, ...]:
    """``pack_specs``' cores only."""
    return pack_specs(specs, core_types, allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                      bounded=bounded)[0]


def unit_tile_count(specs: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], *,
                    allow_coalescing: bool, allow_splitting: bool) -> int:
    """Physical cores one layer's specs pack into alone on the declared core types (the pool unbounded)."""
    return sum(t for _, t in packed_cores(specs, core_types, allow_coalescing=allow_coalescing,
                                          allow_splitting=allow_splitting))
