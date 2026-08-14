"""The thesis' program: ordered passes over the alternating NeuralOps/ComputeOps structure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from mimarsinan.mapping.support.schedule.pass_cut import TRANSFER_DISCIPLINES
from mimarsinan.deployment_record.schema.serde import (
    require_choice,
    strict_kwargs,
    tuple_of,
)

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
    # Executions the wall covers: the divisor that puts a whole-run measured
    # total onto the per-pass (per-sample) axis the latency decomposition uses.
    invocations: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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

    def __post_init__(self) -> None:
        require_choice("SegmentRecord", "pass_reason", self.pass_reason, PASS_REASONS)
        require_choice("SegmentRecord", "programming", self.programming, PROGRAMMING_KINDS)
        if self.pass_index > 0 and self.pass_reason != "capacity_overflow":
            raise ValueError(
                f"SegmentRecord pass_index {self.pass_index} > 0 requires "
                f"pass_reason 'capacity_overflow', got {self.pass_reason!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["bank_ids"] = tuple(int(b) for b in kwargs["bank_ids"])
        kwargs["cores"] = tuple_of(SegmentCoreRecord.from_dict, kwargs["cores"])
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

    A pass boundary is one the schedule INTRODUCES, so the intermediate signal has to
    be buffered either way; ``transfer`` says what was buffered. VERBATIM keeps the
    spike raster and reproduces the fused execution bit-for-bit; COLLAPSE keeps window
    counts and re-emits an even train, which is cheaper and normalizes rhythm. They are
    DIFFERENT computations, so a record that did not name the one it ran would leave a
    reader unable to say what produced the numbers.
    """

    transfer: str
    carried_wires: int
    carried_bytes: int
    peak_live_bytes: int
    timesteps: int

    def __post_init__(self) -> None:
        require_choice("PassCarryRecord", "transfer", self.transfer,
                       frozenset(TRANSFER_DISCIPLINES))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassCarryRecord":
        return cls(**strict_kwargs(cls, data))


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
    # Additive-optional (the ``invocations`` / ``partition`` precedent): a program
    # with no intra-segment pass boundary carries nothing, and pre-carry records
    # load with None rather than forcing a format-version bump.
    carry: Optional["PassCarryRecord"] = None

    def segments(self) -> Tuple[SegmentRecord, ...]:
        return tuple(s for s in self.stages if isinstance(s, SegmentRecord))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stages"] = [_stage_to_dict(stage) for stage in self.stages]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["stages"] = tuple_of(_stage_from_dict, kwargs["stages"])
        carry = kwargs.get("carry")
        if carry is not None and not isinstance(carry, PassCarryRecord):
            kwargs["carry"] = PassCarryRecord.from_dict(carry)
        return cls(**kwargs)
