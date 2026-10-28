"""The pass-period census both pricing planes share, and the residency seam an executed plan marks."""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass, fields
from numbers import Integral
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal

#: One name for a ``HybridStage`` attribute and a ``PassProgram`` field: the frozenset of the
#: pass's hard-core indices that stay programmed across periods.
SCHEDULE_RESIDENT_CORES = "schedule_resident_cores"
#: The ``HybridHardCoreMapping`` attribute holding the executed plan, one ``ResidencySegmentPlan``
#: per neural segment.
SCHEDULE_RESIDENCY_PLAN = "schedule_residency_plan"
#: The flags that mark a whole pass resident: a stage's, then a ``PassProgram``'s.
_WHOLE_PASS_FLAGS = ("schedule_weights_resident", "resident")

#: ``residency_plan.step``: the minimum-r step, the stateless plan, or the step of a declared index.
STEP_FLOOR, STEP_STATELESS, STEP_INDEX = "floor", "stateless", "index"
RESIDENCY_STEP_RULES = (STEP_FLOOR, STEP_STATELESS, STEP_INDEX)
#: ``cores[i].write_granularity``: the unit one write commits.
WRITE_GRANULARITIES = ("cell", "row", "column", "core")
#: ``write_overlap.mode``: writes exposed after compute, or hidden behind it up to the shadow capacity.
WRITE_OVERLAP_MODES = ("serial", "overlapped")
#: N2 ``cores[i].operand.write_source``: payloads written by a host op, or by the producer's on-chip write-back.
OPERAND_WRITE_SOURCES = ("host", "on_chip")
#: N2 ``readout_ranges``: one output range per perceptron (G2's law), or one per consumer group of its columns.
READOUT_RANGE_RULES = ("per_perceptron", "per_consumer_group")


class PeriodCensusError(ToolchainRefusal, ValueError):
    """A period census or a residency mark that does not describe a program."""


def _strict(cls: type, data: Mapping[str, Any]) -> Dict[str, Any]:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise PeriodCensusError(f"{cls.__name__} has unknown fields {sorted(unknown)}")
    return dict(data)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PeriodCensusError(message)


def natural(value: Any, what: str, *, error: type = PeriodCensusError) -> int:
    """The seam's one rule for a count or an index: an integral value of any int type (numpy
    included, bool excluded) that is >= 0, returned as a Python int by ``operator.index`` (exact,
    never truncating); anything else refuses by name with ``error``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or operator.index(value) < 0:
        raise error(f"{what} must be a non-negative integer, got {value!r}")
    return operator.index(value)


def rotation_rewrites(writing_passes: Any) -> bool:
    """The executed rotation, the one period rule of every plane: each execution stage loads its cores from its
    core type's first core, so two or more passes that write displace each other and rewrite their unkept cores
    every period, while a lone writing pass is never displaced and stays loaded across periods."""
    return natural(writing_passes, "writing passes") >= 2


def boundary_crossings(crossings: Iterable[Tuple[int, int, int]],
                       replayed: Iterable[Tuple[int, int]] = ()) -> Tuple[Tuple[int, int, int], ...]:
    """The crossing rule of every plane (record, candidate and plan): per boundary with a carried wire,
    ``(boundary, width read out, width re-injected)`` from ``(width, producing pass, last consuming pass)``. A wire is
    read out after its producing pass and re-injected before its last consuming pass, one crossing per direction
    however many boundaries it lives across (the dense census); boundary ``p`` follows pass ``p``. [U1 W10]
    ``replayed``: ``(width, reading pass)`` per delayed slice beyond one per producer and pass (the builder replays a
    carried producer once per delivery delay), one more re-injection before its reading pass; a plan's are
    ``phase_replays`` [F2]."""
    out: Dict[int, int] = {}
    inbound: Dict[int, int] = {}
    for width, produced, consumed in crossings:
        out[int(produced)] = out.get(int(produced), 0) + int(width)
        inbound[int(consumed) - 1] = inbound.get(int(consumed) - 1, 0) + int(width)
    for width, reader in replayed:
        inbound[int(reader) - 1] = inbound.get(int(reader) - 1, 0) + int(width)
    return tuple((p, out.get(p, 0), inbound.get(p, 0)) for p in sorted(set(out) | set(inbound)))


def phase_replays(delayed_reads: Iterable[Tuple[int, int, int, int, int]], phases: Iterable[Tuple[int, int]]
                  ) -> Tuple[Tuple[int, int], ...]:
    """[F2] A plan's ``replayed`` slices, ``(width, reading pass)``: the units' delayed reads ``(producer width,
    producing unit, reading unit, producer, delay)`` (``pass_timing.delayed_reads``) on the passes' unit ranges, k - 1
    per pass that reads a producer of an earlier pass at k delays, the builder's rule (``project_planned_passes``)."""
    phase_of = {u: q for q, (a, b) in enumerate(phases) for u in range(int(a), int(b))}
    seen: Dict[Tuple[int, int], Tuple[int, set]] = {}
    for width, produced, read, producer, delay in delayed_reads:
        if phase_of[int(produced)] < phase_of[int(read)]:
            seen.setdefault((int(producer), phase_of[int(read)]), (int(width), set()))[1].add(int(delay))
    return tuple((w, q) for (_p, q), (w, delays) in sorted(seen.items()) for _ in range(len(delays) - 1))


def phase_crossings(unit_spans: Iterable[Tuple[int, int, int]], phases: Iterable[Tuple[int, int]],
                    delayed_reads: Iterable[Tuple[int, int, int, int, int]] = ()) -> Tuple[Tuple[int, int], ...]:
    """[W03] Per boundary of a plan's passes, (elements read out, elements re-injected) by ``boundary_crossings``:
    the carried spans on units ``(width, producing unit, last consuming unit)`` mapped onto the passes' unit ranges,
    and [F2] the replays of the units' ``delayed_reads`` (``phase_replays``), as the record and candidate planes."""
    bounds = list(phases)
    phase_of = {u: q for q, (a, b) in enumerate(bounds) for u in range(int(a), int(b))}
    by = {p: (o, i) for p, o, i in boundary_crossings(
        ((w, phase_of[int(p)], phase_of[int(c)]) for w, p, c in unit_spans if phase_of[int(p)] != phase_of[int(c)]),
        phase_replays(delayed_reads, bounds))}
    return tuple(by.get(q, (0, 0)) for q in range(max(0, len(bounds) - 1)))


def settle_naturals(instance: Any, names: Iterable[str], *, error: type = PeriodCensusError) -> None:
    """Hold every named field of a frozen dataclass as ``natural`` of its value (None refuses)."""
    owner = type(instance).__name__
    for name in names:
        object.__setattr__(instance, name, natural(getattr(instance, name), f"{owner}.{name}", error=error))


def settle_residency_segment(segment: Any, *, error: type = PeriodCensusError) -> None:
    """One segment of an executed residency plan (``ResidencySegmentPlan`` or its record): a known step
    rule, a step index exactly when the rule is not stateless, and every count a ``natural``."""
    owner = type(segment).__name__
    if segment.step_rule not in RESIDENCY_STEP_RULES:
        raise error(f"{owner}.step_rule must be one of {sorted(RESIDENCY_STEP_RULES)}, "
                    f"got {segment.step_rule!r}")
    if (segment.step_index is None) != (segment.step_rule == STEP_STATELESS):
        raise error(f"{owner}.step_index {segment.step_index!r}: an index names the executed step "
                    f"exactly when the rule is not {STEP_STATELESS!r} (rule {segment.step_rule!r})")
    stateless = segment.step_index is None
    settle_naturals(segment, [f.name for f in fields(segment) if f.name != "step_rule"
                              and not (stateless and f.name == "step_index")], error=error)


@dataclass(frozen=True)
class PassCensus:
    """One pass of a period: the cores it occupies and keeps, what it writes and what it computes."""

    #: Position in program order across every segment (0, 1, 2, ...). Not ``SegmentRecord.pass_index``,
    #: which restarts at 0 in each segment.
    position: int
    segment_index: int
    #: The whole pass stays programmed across periods (U1's bank-clustered residency).
    resident: bool
    cores: int
    #: Cores of this pass that stay programmed across periods (all of them when ``resident``).
    resident_cores: int
    #: Cells rewritten from the backing store once per period.
    written_cells_backing_store: int
    #: Cells a producer writes once per inference (dynamic operands).
    written_cells_producer: int
    #: Bytes programmed when the pass loads (0 when resident).
    payload_bytes: int
    compute_steps: int
    #: [U-C C-6] Rewritten cores of this pass whose every placement reads a bound bank (N2): they hold no static
    #: program, so they take no per-core programming charge. Serialized only when nonzero.
    operand_only_cores: int = 0

    def __post_init__(self) -> None:
        settle_naturals(self, [f.name for f in fields(self) if f.name != "resident"])
        _require(self.resident_cores <= self.cores,
                 f"PassCensus.resident_cores {self.resident_cores} exceeds cores {self.cores}")
        _require(self.operand_only_cores <= self.cores - self.resident_cores,
                 f"PassCensus.operand_only_cores {self.operand_only_cores} exceeds the rewritten cores "
                 f"{self.cores - self.resident_cores}")
        _require(not self.resident or self.resident_cores == self.cores,
                 "PassCensus.resident_cores must equal cores on a wholly resident pass")


@dataclass(frozen=True)
class PassBoundaryCensus:
    """The carry across the boundary after ``after_pass``, per direction (the carry components price these)."""

    #: ``position`` of the pass the boundary follows (the next pass is ``after_pass + 1``, same segment).
    after_pass: int
    #: Bytes read out of the chip after ``after_pass`` (``carry_readout``).
    carry_out_bytes: int
    #: Bytes injected into the chip before the next pass (``carry_inject``).
    carry_in_bytes: int

    def __post_init__(self) -> None:
        settle_naturals(self, [f.name for f in fields(self)])


@dataclass(frozen=True)
class PassPeriodCensus:
    """Every pass of one period in program order and every intra-segment boundary between them."""

    passes: Tuple[PassCensus, ...]
    boundaries: Tuple[PassBoundaryCensus, ...] = ()
    #: Provenance of the census (which plane or plan step produced it).
    label: str = ""

    def __post_init__(self) -> None:
        for position, census in enumerate(self.passes):
            _require(census.position == position,
                     f"position {census.position} at {position}: passes run in program order "
                     "across segments")
        previous = -1
        for boundary in self.boundaries:
            after = boundary.after_pass
            _require(previous < after < len(self.passes) - 1,
                     f"after_pass {after}: a boundary sits between two passes, in order, once")
            _require(self.passes[after].segment_index == self.passes[after + 1].segment_index,
                     f"after_pass {after}: a pass boundary never crosses a segment (host ops do)")
            previous = after

    @property
    def carry_out_bytes(self) -> int:
        return sum(b.carry_out_bytes for b in self.boundaries)

    @property
    def carry_in_bytes(self) -> int:
        return sum(b.carry_in_bytes for b in self.boundaries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passes": [{k: v for k, v in asdict(p).items() if k != "operand_only_cores" or v} for p in self.passes],
            "boundaries": [asdict(b) for b in self.boundaries],
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassPeriodCensus":
        kwargs = _strict(cls, data)
        kwargs["passes"] = tuple(PassCensus(**_strict(PassCensus, p)) for p in kwargs["passes"])
        kwargs["boundaries"] = tuple(
            PassBoundaryCensus(**_strict(PassBoundaryCensus, b))
            for b in kwargs.get("boundaries", ())
        )
        return cls(**kwargs)


@dataclass(frozen=True)
class ResidencySegmentPlan:
    """One neural segment of an executed residency plan (the record's ``ResidencySegmentRecord`` twin)."""

    segment_index: int
    #: One of ``RESIDENCY_STEP_RULES``.
    step_rule: str
    #: Index of the executed step in the step function; None for the stateless plan.
    step_index: Optional[int]
    #: r: tiles rewritten every period (the pinned budget is core_budget - r).
    rotating_tiles: int
    #: Q: passes of the segment.
    passes: int
    carried_elements: int
    #: W = F - C + r: tiles written per period.
    written_tiles: int
    #: F: tiles of every unit of the program.
    total_tiles: int
    #: C: the program's core budget.
    core_budget: int

    def __post_init__(self) -> None:
        settle_residency_segment(self)


def resident_core_indices(pass_like: Any, core_count: int) -> FrozenSet[int]:
    """The cores of one pass that stay programmed across periods, on either plane.

    Every core when the pass is wholly resident, the declared ``schedule_resident_cores``
    marks when present, else none. ``core_count`` and every mark follow the seam's one integer
    rule (``natural``); a mark set that is not a collection, or a mark outside the pass, refuses
    by name rather than being truncated to a core index.
    """
    count = natural(core_count, "core_count")
    if any(bool(getattr(pass_like, flag, False)) for flag in _WHOLE_PASS_FLAGS):
        return frozenset(range(count))
    marks = getattr(pass_like, SCHEDULE_RESIDENT_CORES, None)
    if marks is None:
        return frozenset()
    _require(isinstance(marks, Iterable) and not isinstance(marks, (str, bytes)),
             f"{SCHEDULE_RESIDENT_CORES} {marks!r} must be a collection of core indices")
    indices = frozenset(natural(i, f"{SCHEDULE_RESIDENT_CORES} mark") for i in marks)
    _require(all(i < count for i in indices),
             f"{SCHEDULE_RESIDENT_CORES} {sorted(indices)} lie outside the pass's {count} cores")
    return indices
