"""The thesis' program: ordered passes over the alternating NeuralOps/ComputeOps structure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.support.period_census import natural, settle_residency_segment
from mimarsinan.mapping.support.schedule.pass_cut import TRANSFER_DISCIPLINES
from mimarsinan.deployment_record.schema.wavefront import WavefrontRecord, settle_core_fields, wavefronts_from
from mimarsinan.deployment_record.schema.serde import (declared_only, require_choice, strict_kwargs, tuple_of,
                                                       without_unset_declared)

PASS_REASONS = frozenset({"initial", "capacity_overflow"})
PROGRAMMING_KINDS = frozenset({"reprogram", "resident"})


@dataclass(frozen=True)
class ComputeOpRecord:
    """One host ComputeOp stage of the program."""

    stage_index: int
    name: str
    op_type: str
    output_width: int
    wall_s_total: Optional[float]
    # Executions (calls) the wall covers.
    invocations: Optional[int] = None
    #: [U-HOST] The samples each timed call carried, as the runner stated them: with the calls, the inferences the wall
    #: covers (``stage_timing.host_wall``, the one per-inference host wall). Unset on an untimed op (its bytes kept).
    samples_per_call: Optional[int] = declared_only()

    def to_dict(self) -> Dict[str, Any]:
        return without_unset_declared(self, asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComputeOpRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class SegmentCoreRecord:
    """One hard core of a neural segment: geometry, occupancy, programmed payload."""

    core_index: int
    axons: int
    neurons: int
    axons_used: int
    neurons_used: int
    cells_used: int
    params_bytes: int
    connectivity_entries: int
    static_delay_levels: Optional[int]
    #: [CL-SG] The used rows a per-axon core's sign register signs (its payload counts one bit each, in
    #: ``params_bytes``); unset on a per-synapse core, so its record keeps its bytes.
    sign_register_rows: Optional[int] = declared_only()
    #: [U-C C-6] On a program with bound banks (N2), whether every placement of the core reads one: such a core holds
    #: no static program and takes no per-core programming charge. Unset on every other program (its bytes kept).
    operand_only: Optional[bool] = declared_only()

    def to_dict(self) -> Dict[str, Any]:
        return without_unset_declared(self, asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentCoreRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class SegmentRecord:
    """One (neural segment × pass) of the schedule."""

    stage_index: int
    segment_index: int
    pass_index: int
    pass_reason: str
    programming: str
    bank_ids: Tuple[int, ...]
    cores: Tuple[SegmentCoreRecord, ...]
    params_programmed: int
    params_unique: int
    params_bytes: int
    connectivity_entries: int
    static_latency_levels: int
    #: Hard cores of this pass an executed residency plan keeps programmed across periods.
    resident_core_indices: Optional[Tuple[int, ...]] = declared_only()
    #: [E-6, N2 F2] Per hard core of this pass (in ``cores`` order), the cells of its bound-bank regions: the operand
    #: payload the producer writes every inference (``mapping.ir.dynamic_bank.operand_cells_by_core``; 0 on a static
    #: core). Sealed on every pass of a program that binds dynamic banks; declared-only, so any other record keeps
    #: its bytes.
    operand_core_cells: Optional[Tuple[int, ...]] = declared_only()
    #: [U-F4] A wavefront group's first pass: its cores no instance of the pass uses (loaded, not evaluated).
    dark_core_indices: Optional[Tuple[int, ...]] = declared_only()
    core_ordinals: Optional[Tuple[int, ...]] = declared_only()  # [U-F4] a later wavefront pass: each core's head core
    dependency_depth: Optional[int] = declared_only()  # [U-LIF2 W7] the pass's level count D (unset: a format-1 record)
    window_arrivals: Optional[int] = declared_only()  # [U-LIF2 r6] neuron-weighted producer windows taken early
    #: [thru:w3] Under the batch pipeline, the input slot whose rows the stage serves (``schedule_input_slot``).
    input_slot: Optional[int] = declared_only()

    def __post_init__(self) -> None:
        require_choice("SegmentRecord", "pass_reason", self.pass_reason, PASS_REASONS)
        dark, ordinals = settle_core_fields("SegmentRecord", len(self.cores), self.dark_core_indices, self.core_ordinals)
        object.__setattr__(self, "dark_core_indices", dark)
        object.__setattr__(self, "core_ordinals", ordinals)
        require_choice("SegmentRecord", "programming", self.programming, PROGRAMMING_KINDS)
        if self.resident_core_indices is not None:
            object.__setattr__(self, "resident_core_indices", _core_indices(self.resident_core_indices))
        if self.operand_core_cells is not None:
            cells = tuple(int(c) for c in self.operand_core_cells)
            if len(cells) != len(self.cores) or any(c < 0 for c in cells):
                raise ValueError(f"SegmentRecord operand_core_cells states {len(cells)} non-negative counts for "
                                 f"{len(self.cores)} cores, one per core required")
            object.__setattr__(self, "operand_core_cells", cells)
        if self.pass_index > 0 and self.pass_reason != "capacity_overflow":
            raise ValueError(
                f"SegmentRecord pass_index {self.pass_index} > 0 requires "
                f"pass_reason 'capacity_overflow', got {self.pass_reason!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        data = without_unset_declared(self, asdict(self))
        # Each core by its own rule, so an unset declared-only core field stays out of the bytes.
        data["cores"] = tuple(core.to_dict() for core in self.cores)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["bank_ids"] = tuple(int(b) for b in kwargs["bank_ids"])
        kwargs["cores"] = tuple_of(SegmentCoreRecord.from_dict, kwargs["cores"])
        if kwargs.get("operand_core_cells") is not None:
            kwargs["operand_core_cells"] = tuple(int(c) for c in kwargs["operand_core_cells"])
        return cls(**kwargs)


ScheduleStage = Union[SegmentRecord, ComputeOpRecord]

_STAGE_KINDS: Dict[str, type] = {
    "segment": SegmentRecord,
    "compute_op": ComputeOpRecord,
}


def _stage_to_dict(stage: ScheduleStage) -> Dict[str, Any]:
    kind = "segment" if isinstance(stage, SegmentRecord) else "compute_op"
    return {"stage_kind": kind, **stage.to_dict()}


def _stage_from_dict(data: Mapping[str, Any]) -> ScheduleStage:
    payload = dict(data)
    kind = payload.pop("stage_kind", None)
    if kind not in _STAGE_KINDS:
        raise ValueError(
            f"schedule stage has unknown stage_kind {kind!r}; "
            f"expected one of {sorted(_STAGE_KINDS)}"
        )
    return _STAGE_KINDS[kind].from_dict(payload)


@dataclass(frozen=True)
class PassCarryRecord:
    """What crossed the INTRA-SEGMENT pass boundaries, and under which discipline.

    A pass boundary is one the schedule INTRODUCES, so the intermediate signal has to be buffered either way;
    ``transfer`` says what was buffered. VERBATIM keeps the spike raster and reproduces the fused execution
    bit-for-bit; COLLAPSE keeps window counts and re-emits an even train, which is cheaper and normalizes rhythm. They
    are DIFFERENT computations, so a record that did not name the one it ran would leave a reader unable to say what
    produced the numbers.
    """

    transfer: str
    carried_wires: int
    carried_bytes: int
    peak_live_bytes: int
    timesteps: int
    #: [E3] What CROSSED the chip boundary, per direction and per inference — a different question from what was
    #: buffered between the passes. Dense under both disciplines (the chip emits and consumes per cycle; the collapse to
    #: counts is host-side), so under COLLAPSE these exceed ``carried_bytes``. Default 0 keeps records sealed before E3
    #: loadable.
    boundary_out_bytes: int = 0
    boundary_in_bytes: int = 0

    def __post_init__(self) -> None:
        require_choice("PassCarryRecord", "transfer", self.transfer,
                       frozenset(TRANSFER_DISCIPLINES))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassCarryRecord":
        return cls(**strict_kwargs(cls, data))


class ResidencyRecordError(ToolchainRefusal, ValueError):
    """A residency record whose step rule, step index, count or core index does not describe an executed plan."""


def _core_indices(marks: Any) -> Tuple[int, ...]:
    """``SegmentRecord.resident_core_indices`` under the seam's one integer rule (``natural``)."""
    what = "SegmentRecord.resident_core_indices"
    if not isinstance(marks, (tuple, list)):
        raise ResidencyRecordError(f"{what} must be a sequence of core indices, got {marks!r}")
    return tuple(natural(index, what, error=ResidencyRecordError) for index in marks)


@dataclass(frozen=True)
class ResidencySegmentRecord:
    """One neural segment of an executed residency plan (field for field ``ResidencySegmentPlan``)."""

    segment_index: int
    step_rule: str
    step_index: Optional[int]
    rotating_tiles: int
    passes: int
    carried_elements: int
    written_tiles: int
    total_tiles: int
    core_budget: int

    def __post_init__(self) -> None:
        settle_residency_segment(self, error=ResidencyRecordError)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResidencySegmentRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class ResidencyPlanRecord:
    """The executed residency plan, one row per neural segment."""

    segments: Tuple[ResidencySegmentRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"segments": [segment.to_dict() for segment in self.segments]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResidencyPlanRecord":
        kwargs = strict_kwargs(cls, data)
        return cls(segments=tuple_of(ResidencySegmentRecord.from_dict, kwargs["segments"]))

    @classmethod
    def from_plan(cls, plan: Any) -> "ResidencyPlanRecord":
        """The record of an executed plan (``ResidencySegmentPlan`` rows), copied field by field."""
        names = [f.name for f in fields(ResidencySegmentRecord)]
        return cls(segments=tuple(
            ResidencySegmentRecord(**{name: getattr(row, name) for name in names}) for row in plan
        ))


@dataclass(frozen=True)
class ScheduleRecord:
    """The deployed program in execution order, with the pass census."""

    stages: Tuple[ScheduleStage, ...]
    pass_count: int
    sync_count: int
    reprogram_passes: int
    reuse_passes: int
    params_reloaded: int
    compute_op_count: int
    # Additive-optional (the ``invocations`` / ``partition`` precedent): a program with no intra-segment pass boundary
    # carries nothing, and pre-carry records load with None rather than forcing a format-version bump.
    carry: Optional["PassCarryRecord"] = None
    #: The executed residency plan; declared-only, so a record without one keeps its bytes.
    residency_plan: Optional[ResidencyPlanRecord] = declared_only()
    #: [KA-G3-1] A value-domain build's NoC wire census (value-hops on the resolved floorplan; the value rule reads
    #: it at ``activation_bits`` as ``noc_value_bytes``); declared-only, so a record without one keeps its bytes.
    value_noc_wire_hops: Optional[int] = declared_only()
    #: [U-F4] The wavefront groups a pipelined program runs; declared-only, so a record without one keeps its bytes.
    wavefronts: Optional[Tuple[WavefrontRecord, ...]] = declared_only()

    def segments(self) -> Tuple[SegmentRecord, ...]:
        return tuple(s for s in self.stages if isinstance(s, SegmentRecord))

    def to_dict(self) -> Dict[str, Any]:
        data = without_unset_declared(self, asdict(self))
        data["stages"] = [_stage_to_dict(stage) for stage in self.stages]
        if self.residency_plan is not None:
            data["residency_plan"] = self.residency_plan.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["stages"] = tuple_of(_stage_from_dict, kwargs["stages"])
        carry = kwargs.get("carry")
        if carry is not None and not isinstance(carry, PassCarryRecord):
            kwargs["carry"] = PassCarryRecord.from_dict(carry)
        plan = kwargs.get("residency_plan")
        if plan is not None and not isinstance(plan, ResidencyPlanRecord):
            kwargs["residency_plan"] = ResidencyPlanRecord.from_dict(plan)
        kwargs["wavefronts"] = wavefronts_from(kwargs.get("wavefronts"))
        return cls(**kwargs)
