"""The pass-period census of both planes (E49): what a schedule period rewrites, per pass and per core.

One rule turns a plane's passes into ``PassPeriodCensus`` rows: the executed rotation
(``period_census.rotation_rewrites``). A pass keeps the cores the residency seam marks (an executed
residency plan) or U1's whole-pass flag keeps; when two or more passes write, each rewrites its other
cores once per period; a lone writing pass is never displaced, so nothing is rewritten. Endurance is a
property of the physical core, so the census also counts the writes each physical core takes
(``SlotWrites``; N2's operand regions by the same rule, ``producer_slots``).

**Deployment model: the program as executed.** The runner loads each execution stage's cores from
its type's first core (R3, ``mapping/noc/fragments.py``), so the j-th rewritten core of a type sits on
the type's j-th unpinned core, and a program of several writing passes rewrites them every period
whether or not its tiles would fit the cores side by side (interaction finding 4, option (a)): a fit
is resident only where an executed residency plan pins it (the floor at F <= C, and [W03] the planner's
fit pins). Operand regions (N2) are written in every pass that holds them, rotating or not.

**Core types.** The rotation needs no core type. A plane that states no core type of its cores on a
chip of several types states every pass's rewrite, placed on no physical core (the lifetime is then
the pigeonhole bound).

**Execution stages.** R3 loads each execution stage from its type's first core, and a pass under
per-hop re-timing runs one stage per depth level. A rewritten row that states several execution
stages (``stage_latencies``) is therefore placed on no physical core (the lifetime is the pigeonhole
bound). A sealed record states one stage per (pass, segment) row and no re-timing, so a re-timed
record is placed per row: a stated boundary until the record seals its execution stages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.support.core_slots import (  # SlotWrites and CoreType re-exported
    CoreTiles, CoreType as CoreType, SlotWrites as SlotWrites, kind_name, place, stated_tiles,
)
from mimarsinan.deployment_record.build.payload_sizes import core_payload_bytes
from mimarsinan.mapping.support.period_census import (
    PassBoundaryCensus, PassCensus, PassPeriodCensus, rotation_rewrites,
)


class WriteCensusError(ToolchainRefusal, ValueError):
    """A period census that cannot be stated: per-core writes no core assignment can hold, a core
    type the chip does not declare, or a chip without a complete core geometry."""


@dataclass(frozen=True)
class PassRow:
    """One pass of one segment as a plane states it: the only input of the census rule."""

    segment_index: int
    #: U1's whole-pass residency flag.
    resident: bool
    core_cells: Tuple[int, ...]
    #: Cores the residency seam keeps across periods (``resident_core_indices``).
    marks: FrozenSet[int]
    #: Used (rows, columns) per core, where the plane states them.
    core_extents: Optional[Tuple[Tuple[int, int], ...]]
    #: Executed steps of one inference of this pass, None where the plane does not state them.
    compute_steps: Optional[int]
    #: The pass's execution-stage latencies (the value rule's window reads them).
    stage_latencies: Optional[Tuple[int, ...]] = None
    #: The core type of each core, where the plane states it (a one-type chip states it for every plane).
    core_types: Optional[Tuple[CoreType, ...]] = None
    #: [CL-SG] Per core, the rows its per-row sign register signs (0 on a per-synapse core); empty: none.
    core_sign_rows: Tuple[int, ...] = ()
    #: N2: payload cells the pass's operand regions take per inference (never in ``core_cells``), and their cores.
    producer_cells: int = 0
    producer_cores: FrozenSet[int] = frozenset()
    #: N2b: the payload cells of each core, where the plane states them (a share per write technology needs them).
    producer_core_cells: Optional[Tuple[int, ...]] = None
    #: [U-C C-6] The cores whose every placement reads a bound bank (no static program), where the plane states them.
    operand_only_cores: FrozenSet[int] = frozenset()
    #: [U-F4] Tiles of a wavefront group's first pass that it loads (writes) but does not evaluate (its dark cores').
    dark: FrozenSet[int] = frozenset()
    #: [U-ENERGY] Per core, the connectivity entries a load programs with it (a sealed record states them); empty: none.
    core_entries: Tuple[int, ...] = ()
    #: [thru:w3] Under the batch pipeline, the input slots whose instances the pass runs; None without it.
    slots: Optional[FrozenSet[int]] = None

    def core_bytes(self, i: int, weight_bits: int) -> int:
        """Core ``i``'s payload by the one rule (``core_payload_bytes``): its cells and its sign register."""
        signs = self.core_sign_rows[i] if i < len(self.core_sign_rows) else 0
        return core_payload_bytes(self.core_cells[i], weight_bits, signs)


#: (after_pass, carried elements read out after it, carried elements re-injected before the next).
BoundaryWidths = Tuple[int, int, int]


@dataclass(frozen=True)
class ProgramPasses:
    """A plane's passes and per-boundary carried widths, before any domain sizes the bytes."""

    rows: Tuple[PassRow, ...]
    boundary_widths: Tuple[BoundaryWidths, ...] = ()
    #: [U-J targeted M2] Per boundary, the widths of ``boundary_widths`` a wavefront group produces and reads itself
    #: (``pass_carry.group_held``), which a declared on-chip carry buffer holds; None: the plane states no split.
    onchip_boundary_widths: Optional[Tuple[BoundaryWidths, ...]] = None
    #: N2: why the plane states no operand regions per pass (None: it states them, or the program has none).
    operands_unstated: Optional[str] = None


def spans_match_carry(passes: ProgramPasses, carry: Optional[Mapping[str, Any]], timesteps: Optional[int]) -> bool:
    """Whether the per-boundary widths (one crossing per direction per carried wire) account for
    every train the plane's carry census counts; a census that counts more (a delayed slice's extra
    re-injection) states its totals, never a per-boundary split the spans cannot attribute.
    ([U-C4 targeted] moved here unchanged from ``quantities/write_law.py``, which re-exports it.)"""
    if carry is None or timesteps is None:
        return True
    per = (int(timesteps) + 7) // 8
    out = sum(w for _, w, _ in passes.boundary_widths) * per
    inbound = sum(w for _, _, w in passes.boundary_widths) * per
    return (out, inbound) == (int(carry["boundary_out_bytes"]), int(carry["boundary_in_bytes"]))


@dataclass(frozen=True)
class WriteCensus:
    """A plane's ``PassPeriodCensus`` with what the write pricer reads beside it."""

    census: PassPeriodCensus
    #: Per pass, the (rows, columns) of every rewritten core, or None where the plane has none.
    written_extents: Optional[Tuple[Tuple[Tuple[int, int], ...], ...]]
    #: Whether every pass's compute steps are stated (the overlapped write term needs them).
    compute_sealed: bool
    #: Per core type, the writes per period of each physical core (empty when nothing is rewritten); None where
    #: the census places no rewritten core: an untyped plane on a chip of several core types whose tiles
    #: exceed every declared core, or a rewritten row of several execution stages.
    slot_writes: Optional[Tuple[SlotWrites, ...]] = None
    #: Per pass, the cores of each type it computes on (the overlapped mode's shadow room is per type); None where
    #: the plane states no core type on a chip of several types.
    pass_core_types: Optional[Tuple[Tuple[Tuple[CoreType, int], ...], ...]] = None
    #: N2: per core holding operand regions, (static writes per period, producer writes per inference), distinct
    #: and sorted, by the same slot rule; () without operand regions, None where no rewritten core is placed.
    producer_slots: Optional[Tuple[Tuple[int, int], ...]] = ()
    #: N2b: per pass, the rewritten cores (row indices), and per core type its own ``producer_slots``.
    written_cores: Tuple[Tuple[int, ...], ...] = ()
    type_producer_slots: Optional[Tuple[Tuple[CoreType, Tuple[Tuple[int, int], ...]], ...]] = ()

    @property
    def most_written_core(self) -> Optional[int]:
        """The writes per period of the most-written physical core, or None when unstated."""
        if self.slot_writes is None:
            return None
        return max((w for slot in self.slot_writes for w in slot.writes), default=0)


def written_cores(census: Any) -> int:
    """W: the core writes of one period (every rewritten core of every pass), the pigeonhole bound's count."""
    return sum(p.cores - p.resident_cores for p in census.passes if p.written_cells_backing_store)


def pass_programmed_cores(p: Any) -> int:
    """[U-C2] One pass's share of ``programmed_cores``: its rewritten cores less its rewritten operand-only cores."""
    return p.cores - p.resident_cores - p.operand_only_cores if p.written_cells_backing_store else 0


def programmed_cores(census: Any) -> int:
    """[consts, U-C C-6] e_core_program's count: the rewritten cores whose static program a period writes, W less the
    rewritten operand-only cores (every placement a bound bank, whose operand writes the operand law prices)."""
    return sum(pass_programmed_cores(p) for p in census.passes)


def programmed_basis(census: Any) -> str:
    """The multiplicand ``programmed_cores`` names in a priced basis (the consts name where no core is operand-only)."""
    only = any(p.operand_only_cores for p in census.passes if p.written_cells_backing_store)
    return "rewritten cores holding a static program" if only else "rewritten_cores"


def stage_latency(delays: Sequence[Optional[int]]) -> Optional[int]:
    """A stage's latency span (the max less the min of its cores' latencies), or None when any core
    states none: the window rule's L_s on every plane."""
    stated = [int(d) for d in delays if d is not None]
    if not delays or len(stated) != len(delays):
        return None
    return max(stated) - min(stated)


def core_capacity_of(platform: Mapping[str, Any]) -> Optional[Dict[CoreType, int]]:
    """The declared cores of each physical core type (types of one geometry pooled), or None when a
    declared type states no geometry."""
    cores = list(platform.get("cores") or ())
    if not cores or any("max_axons" not in ct or "max_neurons" not in ct for ct in cores):
        return None
    capacity: Dict[CoreType, int] = {}
    for ct in cores:
        kind = (int(ct["max_axons"]), int(ct["max_neurons"]))
        capacity[kind] = capacity.get(kind, 0) + int(ct.get("count", 1))
    return capacity


def _row_tiles(rows: Sequence[PassRow], capacity: Mapping[CoreType, int]) -> Optional[List[CoreTiles]]:
    """Every row's per-core declared tiles (``core_slots.stated_tiles``), or None when a row states no core type on a
    chip of several types; a core that is no declared core and no fused one refuses by name."""
    only = next(iter(capacity)) if len(capacity) == 1 else None
    typed: List[CoreTiles] = []
    for row in rows:
        if row.core_types is None:
            if only is None:
                return None
            typed.append(((only,),) * len(row.core_cells))
            continue
        tiles = [stated_tiles(kind, capacity) for kind in row.core_types]
        stray = sorted({kind for kind, t in zip(row.core_types, tiles) if t is None})
        if stray:
            raise WriteCensusError(
                f"the plane places cores of type {', '.join(map(kind_name, stray))}, which the chip does not "
                f"declare and which fuse no declared type (declared: {', '.join(map(kind_name, sorted(capacity)))})")
        typed.append(tuple(t for t in tiles if t is not None))
    return typed


def period_census(passes: ProgramPasses, *, weight_bits: int,
                  capacity: Optional[Mapping[CoreType, int]],
                  bytes_of: Optional[Callable[[int], int]] = None, label: str = "",
                  every_core: bool = False) -> WriteCensus:
    """The steady-state period census of a pass list (the one rule, both planes: the executed rotation).

    ``capacity`` (the declared cores per core type) places each rewritten core on a physical core.
    Rows that state no core types on a chip of several types are rewritten by the same rule, placed on
    no physical core (said in the label). ``bytes_of`` sizes the carried elements of a boundary in
    this run's domain; without it the census states no boundary. A pass is ``resident`` exactly when it
    writes nothing per period. [consts] ``every_core``: the deployment load instead, every core of every pass
    written once from an empty chip (no residency flag or mark keeps one), the tile writes of law.md's Pi.
    """
    if not capacity:
        raise WriteCensusError("the platform declares no complete core geometry (max_axons and max_neurons "
                               "of every core type), so no rewritten core can be placed on a declared core")
    rows = passes.rows
    if every_core:
        rows = tuple(replace(row, resident=False, marks=frozenset()) for row in rows)
        label = (f"{label}; " if label else "") + "the deployment load: every core of every pass written once"
    typed = _row_tiles(rows, capacity)
    unkept = [[i for i in range(len(row.core_cells)) if not row.resident and i not in row.marks] for row in rows]
    rotates = every_core or rotation_rewrites(sum(1 for cores in unkept if cores))
    written_cores: List[List[int]] = unkept if rotates else [[] for _ in rows]
    if typed is None and rotates:
        label = (f"{label}; " if label else "") + (
            f"the plane states no core type of its cores on a chip of {len(capacity)} core types: its passes "
            "rewrite, and no rewritten core is placed on a physical core")
    out: List[PassCensus] = []
    extents: List[Tuple[Tuple[int, int], ...]] = []
    for position, row in enumerate(rows):
        written = written_cores[position]
        out.append(PassCensus(
            position=position, segment_index=row.segment_index, resident=not written,
            cores=len(row.core_cells), resident_cores=len(row.core_cells) - len(written),
            written_cells_backing_store=sum(row.core_cells[i] for i in written),
            written_cells_producer=row.producer_cells,
            payload_bytes=sum(row.core_bytes(i, weight_bits) for i in written),
            compute_steps=row.compute_steps or 0,
            operand_only_cores=len(set(written) & row.operand_only_cores),
            connectivity_entries=sum(row.core_entries[i] for i in written)  # [U-ENERGY] the written cores' own
            if len(row.core_entries) == len(row.core_cells) else None,
        ))
        if row.core_extents is not None:
            extents.append(tuple(row.core_extents[i] for i in written))
    # Every pass loads its unkept cores beside every pin, rotating or not: the fit reads them all; the slot rule
    # counts the rewritten ones (N2: static cells and operand regions apart).
    slots: Optional[Tuple[SlotWrites, ...]] = None
    typed_slots = None
    if typed is not None:
        placed, typed_slots, unplaced, unfit = place(
            [row.resident for row in rows], [sorted(row.marks) for row in rows], typed, unkept, capacity,
            written=written_cores, producers=[row.producer_cores for row in rows],
            static=[frozenset(i for i, cells in enumerate(row.core_cells) if cells or i not in row.producer_cores)
                    for row in rows])
        if unfit is not None:
            raise WriteCensusError(unfit)
        if unplaced is not None and rotates:
            label = (f"{label}; " if label else "") + unplaced
        slots = placed
    if not rotates:
        slots = ()
    if any(row.producer_cells and not row.producer_cores for row in rows):
        typed_slots = None  # operand regions on cores the plane does not state: their slots are not placed (N2)
    producer = None if typed_slots is None else tuple(sorted({pair for _, pairs in typed_slots for pair in pairs}))
    staged = [position for position, row in enumerate(rows)
              if written_cores[position] and len(row.stage_latencies or ()) > 1]
    if staged:
        label = (f"{label}; " if label else "") + (
            f"rewritten passes {staged} run several execution stages (per-hop re-timing), each loaded from "
            "its type's first core: no rewritten core is placed on a physical core")
    boundaries = () if bytes_of is None else tuple(
        PassBoundaryCensus(after_pass=after, carry_out_bytes=bytes_of(w_out), carry_in_bytes=bytes_of(w_in))
        for after, w_out, w_in in passes.boundary_widths
    )
    return WriteCensus(
        census=PassPeriodCensus(passes=tuple(out), boundaries=boundaries, label=label),
        written_extents=tuple(extents) if len(extents) == len(rows) else None,
        compute_sealed=all(row.compute_steps is not None for row in rows),
        slot_writes=None if staged else slots,
        pass_core_types=None if typed is None else tuple(
            tuple(sorted(Counter(t for tiles in cores for t in tiles).items())) for cores in typed),
        producer_slots=None if staged else producer,
        written_cores=tuple(tuple(cores) for cores in written_cores),
        type_producer_slots=None if staged else typed_slots,
    )
