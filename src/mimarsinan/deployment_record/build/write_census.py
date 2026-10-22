"""The pass-period census of both planes (E49): what a schedule period rewrites, per pass and per core.

One rule turns a plane's passes into ``PassPeriodCensus`` rows: a program whose programmed tiles fit
the declared cores of their own core type is loaded once (every pass stays programmed across
periods); otherwise each reprogram pass rewrites, once per period, the cores the residency seam does
not keep. Endurance is a property of the physical core, so the census also counts the writes each
physical core takes per period (``SlotWrites``).

**Declared deployment model.** The census prices a loader that keeps a program whose tiles fit
their core types resident, each tile on its own core (disjoint cores, nothing rewritten per period).
A program that does not fit runs the R3 rotation of ``mapping/noc/fragments.py``: each execution
stage's cores load from its type's first core, so the j-th rewritten core of a type sits on the
type's j-th unpinned core. Under R3 alone a fitting multi-segment program would also reload core 0
every period; the zero write terms of a fit are the resident loader's, a stated boundary.

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
from fractions import Fraction
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
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


@dataclass(frozen=True)
class SlotWrites:
    """One core type's period writes per physical core, most-written first.

    The runner loads each pass's cores from the first core of their type (``fragments.py`` R3), so
    the j-th core of a type that a pass rewrites sits on that type's j-th core not pinned by a
    residency mark: core j takes one write from every pass that rewrites more than j cores of the
    type, and the first core one from every pass that rewrites any.
    """

    core_type: CoreType
    writes: Tuple[int, ...]


#: (after_pass, carried elements read out after it, carried elements re-injected before the next).
BoundaryWidths = Tuple[int, int, int]


@dataclass(frozen=True)
class ProgramPasses:
    """A plane's passes and per-boundary carried widths, before any domain sizes the bytes."""

    rows: Tuple[PassRow, ...]
    boundary_widths: Tuple[BoundaryWidths, ...] = ()


@dataclass(frozen=True)
class WriteCensus:
    """A plane's ``PassPeriodCensus`` with what the write pricer reads beside it."""

    census: PassPeriodCensus
    #: Per pass, the (rows, columns) of every rewritten core, or None where the plane has none.
    written_extents: Optional[Tuple[Tuple[Tuple[int, int], ...], ...]]
    #: Whether every pass's compute steps are stated (the overlapped write term needs them).
    compute_sealed: bool
    #: Per core type, the writes per period of each physical core (empty when nothing is rewritten);
    #: None where the census places no rewritten core: an untyped plane on a chip of several core
    #: types whose tiles exceed every declared core, or a rewritten row of several execution stages.
    slot_writes: Optional[Tuple[SlotWrites, ...]] = None
    #: Per pass, the cores of each type it computes on (the overlapped mode's shadow room is per type);
    #: None where the plane states no core type on a chip of several types.
    pass_core_types: Optional[Tuple[Tuple[Tuple[CoreType, int], ...], ...]] = None

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


def shadow_room_refusal(capacity: Optional[Sequence[Tuple[CoreType, int]]], fraction: float,
                        pass_types: Optional[Sequence[Sequence[Tuple[CoreType, int]]]],
                        pass_cores: Sequence[int]) -> Optional[str]:
    """Why overlapped writes cannot hide behind compute, or None. While a pass computes, the next pass's
    tiles are written into the shadow share s of each core type's declared cores, so every pass computes
    on at most floor(count_t x (1 - s)) cores of each type t (s read exactly as declared)."""
    declared = dict(capacity or ())
    if not declared:
        return "overlapped writes keep a shadow share of each core type, and the chip declares no core types"
    if pass_types is None and len(declared) > 1:
        return (f"overlapped writes keep a shadow share of each core type, and the plane states no core type of "
                f"its cores on a chip of {len(declared)} core types: the shadow room is undecidable")
    if pass_types is None:
        pass_types = [((next(iter(declared)), cores),) for cores in pass_cores]
    for kind, count in sorted(declared.items()):
        room = int(count * (1 - Fraction(repr(float(fraction)))))
        widest = max((dict(types).get(kind, 0) for types in pass_types), default=0)
        if widest > room:
            return (f"overlapped writes keep shadow_capacity_fraction {fraction} of each core type's declared "
                    f"cores: a pass computes on {widest} cores of type {_kind(kind)}, more than the {room} of its "
                    f"{count} left to compute (the program must be planned at count_t x (1 - s))")
    return None


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


def _slot_writes(rows: Sequence[PassRow], typed: Sequence[Tuple[CoreType, ...]],
                 written: Sequence[Sequence[int]], capacity: Mapping[CoreType, int]) -> Tuple[SlotWrites, ...]:
    """Per core type, the writes each free physical core takes per period (``SlotWrites``)."""
    pinned: Counter = Counter()
    per_pass: List[Counter] = []
    for row, types, cores in zip(rows, typed, written):
        if not row.resident:
            pinned.update(types[i] for i in row.marks if i < len(types))
        per_pass.append(Counter(types[i] for i in cores))
    out: List[SlotWrites] = []
    for kind in sorted(capacity):
        widest = max((count[kind] for count in per_pass), default=0)
        free = capacity[kind] - pinned[kind]
        if widest > free:
            raise WriteCensusError(
                f"the residency marks pin {pinned[kind]} of the {capacity[kind]} cores of type "
                f"{kind[0]} x {kind[1]}, leaving {free}, and a pass rewrites {widest} of them")
        if widest:
            out.append(SlotWrites(kind, tuple(sum(1 for count in per_pass if count[kind] > j)
                                              for j in range(widest))))
    return tuple(out)


def period_census(passes: ProgramPasses, *, weight_bits: int,
                  capacity: Optional[Mapping[CoreType, int]],
                  bytes_of: Optional[Callable[[int], int]] = None, label: str = "") -> WriteCensus:
    """The steady-state period census of a pass list (the one rule, both planes).

    ``capacity`` (the declared cores per core type) decides loaded-once per type and places each
    rewritten core on a physical core. Rows that state no core types on a chip of several types
    prove a rewrite only when their tiles exceed every declared core (said in the label; no core is
    placed); otherwise loaded-once is undecidable and the census refuses by name. ``bytes_of`` sizes
    the carried elements of a boundary in this run's domain; without it the census states no boundary.
    """
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
            written_cells_producer=0,
            payload_bytes=sum(params_bytes(row.core_cells[i], weight_bits) for i in written),
            compute_steps=row.compute_steps or 0,
        ))
        if row.core_extents is not None:
            extents.append(tuple(row.core_extents[i] for i in written))
    slots = None if typed is None else _slot_writes(rows, typed, written_cores, capacity)
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
    )
