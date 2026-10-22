"""[E85] The cores a set of layout specs packs into, and the declared physical tiles each packed core occupies."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError

#: One packed core: its occupied cells (used axons x used neurons, the rectangle the record's ``params_bytes``
#: prices) and the physical tiles it occupies (a fused core counts its components).
PackedCore = Tuple[int, int]
#: The declared physical cores one packed core occupies, (axons, neurons) per component.
CoreTiles = Tuple[Tuple[int, int], ...]


def core_type_tiles(axons: int, neurons: int, core_types: Sequence[LayoutHardCoreType],
                    components: Optional[Sequence[int]] = None) -> Tuple[Tuple[int, int], ...]:
    """The declared physical cores one packed core occupies: itself, or a fused core's components (axons summed
    over cores of one type, ``canonical_fuse_hardcores``), read from ``components`` or decomposed uniquely."""
    if components:
        return tuple((int(a), int(neurons)) for a in components)
    declared = {(int(t.max_axons), int(t.max_neurons)) for t in core_types}
    if (int(axons), int(neurons)) in declared:
        return ((int(axons), int(neurons)),)
    fits = sorted(a for a, n in declared if n == int(neurons) and int(axons) % a == 0)
    if len(fits) != 1:
        raise ResidencyPlanError(
            f"residency_plan: a fused {axons}x{neurons} core decomposes into the declared core types "
            f"{sorted(declared)} in {len(fits)} ways, so its physical tiles are not determined")
    return ((fits[0], int(neurons)),) * (int(axons) // fits[0])


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
