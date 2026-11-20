"""[E2] One pass's install cost as a candidate's NoC fragments state it (``fragments.collect_noc_fragments``): per
core its committed rectangle and geometry and per physical tile its cells and signed rows, with the residency law's
answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Mapping, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.packing.residency_tiles import CoreTiles, core_type_tiles, tile_extents
from mimarsinan.mapping.platform.platform_constraints import carries_sign_register


@dataclass(frozen=True)
class PassProgram:
    """[E2] One pass's install cost: per-core committed rectangles + the
    residency law's answer (a resident pass installs nothing)."""

    core_cells: Tuple[int, ...]
    resident: bool
    #: [E85] Indices of this pass's cores pinned across periods by an executed residency plan; None: no plan.
    schedule_resident_cores: Optional[FrozenSet[int]] = None
    #: [CL-SG] Per core, the used rows its per-row sign register signs (0 on a per-synapse core); empty: none.
    core_sign_rows: Tuple[int, ...] = ()
    #: Per core, the cells of each physical tile it occupies (a fused core: one per member, ``tile_extents``);
    #: empty: every core is one tile.
    core_tile_cells: Tuple[Tuple[int, ...], ...] = ()
    #: [CL-SG, W04] Per core, the rows each physical tile's sign register signs (a fused core's used rows fill its
    #: members in order, ``tile_extents``, and each member signs its own); empty: none.
    core_tile_sign_rows: Tuple[Tuple[int, ...], ...] = ()
    #: [W03] Per core, the (axons, neurons) it is packed at (a fused instance: its components summed), which the
    #: period census types the cores by; None where a caller states no geometry.
    core_geometry: Optional[Tuple[Tuple[int, int], ...]] = None
    #: [U-F4] A wavefront pass (one evaluation: its cores read carried inputs only), the cores a group's first pass
    #: loads unused (dark) and a later pass's head ordinals; defaults: none of them.
    wavefront: bool = False
    dark_cores: Optional[FrozenSet[int]] = None
    core_ordinals: Optional[Tuple[int, ...]] = None
    #: [U-ENERGY] Per core, its used (rows, columns): the census of outputs per evaluation and of local memory
    #: traffic; empty where a caller states none (those counts are then not produced).
    core_used: Tuple[Tuple[int, int], ...] = ()

    @property
    def cores(self) -> int:
        """Allocated crossbars (a fused core once)."""
        return len(self.core_cells)

    @property
    def tiles(self) -> int:
        """Physical cores (a fused core counts its members): what every priced core count reads."""
        return sum(map(len, self.core_tile_cells)) if self.core_tile_cells else self.cores

    def evaluated(self) -> Tuple[int, ...]:
        """The cores this pass evaluates: every core but its dark ones."""
        return tuple(i for i in range(self.cores) if i not in (self.dark_cores or ()))

    def core_tiles(self, i: int) -> int:
        return len(self.core_tile_cells[i]) if self.core_tile_cells else 1

    @property
    def tile_cells(self) -> Tuple[int, ...]:
        return tuple(c for cells in self.core_tile_cells for c in cells) if self.core_tile_cells else self.core_cells


def packed_pass_program(snapshots: Sequence[Any], *, resident: bool, core_types: Sequence[LayoutHardCoreType],
                        declared_cores: Sequence[Mapping[str, Any]],
                        operand_axons: Sequence[int] = ()) -> Tuple[PassProgram, Tuple[CoreTiles, ...]]:
    """One packed pass's program from the layout packer's used-core snapshots, and each core's physical tiles (a
    fused core's members, ``core_type_tiles``). [CL-SG] ``declared_cores`` (the platform's ``cores``) signs each
    core of a per-axon type (``carries_sign_register``): its used rows, and per tile the rows the tile holds.
    [U-E2 H-1] ``operand_axons`` (per core, the fused-add operand rows placed on it) are axons but no used row,
    cell or programmed byte, as ``CoreOccupancy`` reads a built core."""
    snapshots = tuple(snapshots)
    used = [int(snap.used_axons) - (int(operand_axons[i]) if operand_axons else 0) for i, snap in enumerate(snapshots)]
    tiles = tuple(core_type_tiles(snap.axons_per_core, snap.neurons_per_core, core_types, snap.component_axons)
                  for snap in snapshots)
    extents = [tile_extents(a, snap.used_neurons, t) for a, snap, t in zip(used, snapshots, tiles)]
    signed = [bool(declared_cores) and carries_sign_register(snap.axons_per_core, snap.neurons_per_core,
                                                             declared_cores) for snap in snapshots]
    return PassProgram(
        core_cells=tuple(a * int(snap.used_neurons) for a, snap in zip(used, snapshots)),
        resident=bool(resident),
        core_sign_rows=tuple(a if s else 0 for a, s in zip(used, signed))
        if declared_cores else (),
        core_tile_cells=tuple(tuple(rows * cols for rows, cols in e) for e in extents),
        core_tile_sign_rows=tuple(tuple(rows if s else 0 for rows, _ in e) for e, s in zip(extents, signed))
        if declared_cores else (),
        core_geometry=tuple((int(snap.axons_per_core), int(snap.neurons_per_core)) for snap in snapshots),
        core_used=tuple((int(a), int(snap.used_neurons)) for a, snap in zip(used, snapshots)),
    ), tiles
