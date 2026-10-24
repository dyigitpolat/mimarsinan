"""The pass-period census of both planes (E49): what a schedule period rewrites, per pass and per core.

One rule turns a plane's passes into ``PassPeriodCensus`` rows: a program whose programmed tiles fit
the declared cores of their own core type is loaded once (every pass stays programmed across
periods); otherwise each reprogram pass rewrites, once per period, the cores the residency seam does
not keep. Endurance is a property of the physical core, so the census also counts the writes each
physical core takes (``SlotWrites``; N2's operand regions by the same rule, ``producer_slots``).

**Declared deployment model.** The census prices a loader that keeps a program whose tiles fit
their core types resident, each tile on its own core (disjoint cores, nothing rewritten per period).
A program that does not fit runs the R3 rotation of ``mapping/noc/fragments.py``: each execution
stage's cores load from its type's first core, so the j-th rewritten core of a type sits on the
type's j-th unpinned core. Under R3 alone a fitting multi-segment program would also reload core 0
every period; the zero static write terms of a fit are the resident loader's, a stated boundary.

**Core types.** Loaded once is decided per core type, never on the planner's 0.8x budget. A plane
that states no core type of its cores on a chip of several types proves only what the total core
count proves: tiles beyond every declared core rewrite (exactly as typed: every reprogram pass's
unkept cores), placed on no physical core; tiles within it are undecidable and refuse by name.

**Execution stages.** R3 loads each execution stage from its type's first core, and a pass under
per-hop re-timing runs one stage per depth level. A rewritten row that states several execution
stages (``stage_latencies``) is therefore placed on no physical core (the lifetime is the pigeonhole
bound). A sealed record states one stage per (pass, segment) row and no re-timing, so a re-timed
record is placed per row: a stated boundary until the record seals its execution stages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.payload_sizes import core_payload_bytes
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus


#: A physical core type, ``(axons, neurons)``: the geometry a core is declared and built at.
CoreType = Tuple[int, int]


class WriteCensusError(ToolchainRefusal, ValueError):
    """A period census that cannot be stated: per-core writes no core assignment can hold, a core
    type the chip does not declare, or a loaded-once decision the plane's core types leave open."""


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

    def core_bytes(self, i: int, weight_bits: int) -> int:
        """Core ``i``'s payload by the one rule (``core_payload_bytes``): its cells and its sign register."""
        signs = self.core_sign_rows[i] if i < len(self.core_sign_rows) else 0
        return core_payload_bytes(self.core_cells[i], weight_bits, signs)


@dataclass(frozen=True)
class SlotWrites:
    """One core type's static writes per period per physical core. The runner loads each pass's
    cores starting at the first core of their type (``fragments.py`` R3), so the j-th core of a type
    that a pass rewrites sits on that type's j-th core not pinned by a residency mark: core j takes
    one write per period of every pass whose j-th rewritten core of the type holds static cells."""

    core_type: CoreType
    writes: Tuple[int, ...]


#: (after_pass, carried elements read out after it, carried elements re-injected before the next).
BoundaryWidths = Tuple[int, int, int]


@dataclass(frozen=True)
class ProgramPasses:
    """A plane's passes and per-boundary carried widths, before any domain sizes the bytes."""

    rows: Tuple[PassRow, ...]
    boundary_widths: Tuple[BoundaryWidths, ...] = ()
    #: N2: why the plane states no operand regions per pass (None: it states them, or the program has none).
    operands_unstated: Optional[str] = None


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


def _kind(kind: CoreType) -> str:
    return f"{kind[0]} x {kind[1]}"


def _row_types(rows: Sequence[PassRow], capacity: Mapping[CoreType, int]
               ) -> Optional[List[Tuple[CoreType, ...]]]:
    """Every row's per-core types, or None when a row states none on a chip of several types; a
    row on a core type the chip does not declare refuses by name."""
    only = next(iter(capacity)) if len(capacity) == 1 else None
    typed: List[Tuple[CoreType, ...]] = []
    for row in rows:
        if row.core_types is None:
            if only is None:
                return None
            typed.append((only,) * len(row.core_cells))
            continue
        stray = sorted({kind for kind in row.core_types if kind not in capacity})
        if stray:
            raise WriteCensusError(
                f"the plane places cores of type {', '.join(map(_kind, stray))}, which the chip does not "
                f"declare (declared: {', '.join(map(_kind, sorted(capacity)))})")
        typed.append(tuple(row.core_types))
    return typed


def _slot_writes(rows: Sequence[PassRow], typed: Sequence[Tuple[CoreType, ...]], written: Sequence[Sequence[int]],
                 capacity: Mapping[CoreType, int]) -> Tuple[Tuple[SlotWrites, ...], Tuple[Tuple[CoreType, Any], ...]]:
    """(``SlotWrites`` per core type; per core type, per core holding operand regions, (static writes per period,
    producer writes per inference)). One rule: the j-th rewritten core of a type sits on its j-th free core, and a
    pass writes its operand regions once per sample, so a slot takes one producer write per inference from each pass
    whose j-th core holds them (N2); a kept core (loaded once, resident, marked) with them is its own core."""
    pinned: Counter = Counter()
    static: Dict[CoreType, Counter] = {}
    producer: Dict[CoreType, Counter] = {}
    widest: Counter = Counter()
    dedicated: set = set()
    for row, types, cores in zip(rows, typed, written):
        if not row.resident:
            pinned.update(types[i] for i in row.marks if i < len(types))
        rank: Counter = Counter()
        for i in cores:
            kind, j = types[i], rank[types[i]]
            rank[kind] += 1
            if i not in row.producer_cores or row.core_cells[i]:
                static.setdefault(kind, Counter())[j] += 1
            if i in row.producer_cores:
                producer.setdefault(kind, Counter())[j] += 1
        widest |= rank
        dedicated |= {types[i] for i in set(row.producer_cores) - set(cores)}
    out: List[SlotWrites] = []
    for kind in sorted(capacity):
        free = capacity[kind] - pinned[kind]
        if widest[kind] > free:
            raise WriteCensusError(
                f"the residency marks pin {pinned[kind]} of the {capacity[kind]} cores of type "
                f"{kind[0]} x {kind[1]}, leaving {free}, and a pass rewrites {widest[kind]} of them")
        if widest[kind]:
            out.append(SlotWrites(kind, tuple(static.get(kind, Counter())[j] for j in range(widest[kind]))))
    pairs = {kind: {(static.get(kind, Counter())[j], n) for j, n in producer.get(kind, Counter()).items()}
             | ({(0, 1)} if kind in dedicated else set()) for kind in sorted(set(producer) | dedicated)}
    return tuple(out), tuple((kind, tuple(sorted(slots))) for kind, slots in pairs.items())


def period_census(passes: ProgramPasses, *, weight_bits: int,
                  capacity: Optional[Mapping[CoreType, int]],
                  bytes_of: Optional[Callable[[int], int]] = None, label: str = "") -> WriteCensus:
    """The steady-state period census of a pass list (the one rule, both planes).

    ``capacity`` (the declared cores per core type) decides loaded-once per type and places each
    rewritten core on a physical core. Rows that state no core types on a chip of several types
    prove a rewrite only when their tiles exceed every declared core (said in the label; no core is
    placed); otherwise loaded-once is undecidable and the census refuses by name. ``bytes_of`` sizes
    the carried elements of a boundary in this run's domain; without it the census states no boundary."""
    if not capacity:
        raise WriteCensusError("the platform declares no complete core geometry (max_axons and max_neurons "
                               "of every core type), so no core type is known to hold the program's tiles")
    rows = passes.rows
    typed = _row_types(rows, capacity)
    if typed is not None:
        programmed = Counter(kind for row, types in zip(rows, typed) if not row.resident for kind in types)
        loaded_once = all(count <= capacity[kind] for kind, count in programmed.items())
    else:
        tiles = sum(len(row.core_cells) for row in rows if not row.resident)
        cores = sum(capacity.values())
        untyped = f"the plane states no core type of its cores on a chip of {len(capacity)} core types"
        if tiles <= cores:
            raise WriteCensusError(
                f"{untyped}, and its {tiles} programmed tiles fit the chip's {cores} cores: whether each core "
                "type holds its own tiles (loaded once) is undecidable until the pass program states each "
                "core's type (PassProgram.core_geometry)")
        loaded_once = False
        label = (f"{label}; " if label else "") + (
            f"{untyped}: its {tiles} programmed tiles exceed the chip's {cores} cores, so it rewrites, "
            "and no rewritten core is placed on a physical core")
    out: List[PassCensus] = []
    written_cores: List[List[int]] = []
    extents: List[Tuple[Tuple[int, int], ...]] = []
    for position, row in enumerate(rows):
        kept = frozenset(range(len(row.core_cells))) if loaded_once or row.resident else row.marks
        written = [i for i in range(len(row.core_cells)) if i not in kept]
        written_cores.append(written)
        out.append(PassCensus(
            position=position, segment_index=row.segment_index,
            resident=not written and (loaded_once or row.resident),
            cores=len(row.core_cells), resident_cores=len(row.core_cells) - len(written),
            written_cells_backing_store=sum(row.core_cells[i] for i in written),
            written_cells_producer=row.producer_cells,
            payload_bytes=sum(row.core_bytes(i, weight_bits) for i in written),
            compute_steps=row.compute_steps or 0,
        ))
        if row.core_extents is not None:
            extents.append(tuple(row.core_extents[i] for i in written))
    slots, typed_slots = (None, None) if typed is None else _slot_writes(rows, typed, written_cores, capacity)
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
        pass_core_types=None if typed is None else tuple(tuple(sorted(Counter(types).items())) for types in typed),
        producer_slots=None if staged else producer,
        written_cores=tuple(tuple(cores) for cores in written_cores),
        type_producer_slots=None if staged else typed_slots,
    )
