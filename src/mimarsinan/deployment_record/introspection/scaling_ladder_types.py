"""The scaling ladder's value types and its refusals by name.

Refusals: ``LadderGeometryError`` (the census needs an on-chip softcore, and the
packed tile count needs one declared core shape and a program that packs on
unbounded cores), ``LadderDeclarationError`` (a
row's platform differs from its priced program in more than K, rate and lifetime)
and ``LadderScheduleRefusal``, which a row records by name when the builder's
planner finds no schedule at the point's core budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.schema.provenance import Band


class LadderGeometryError(ToolchainRefusal, ValueError):
    """The census needs an on-chip softcore; the packed tile count needs one declared core shape and a
    program that packs on unbounded cores."""


class LadderDeclarationError(ToolchainRefusal, ValueError):
    """A row's platform differs from its priced program in more than K, rate and lifetime."""


class LadderScheduleRefusal(ToolchainRefusal, ValueError):
    """The builder's planner finds no schedule at the point's core budget (a row records it by name)."""


#: Declarations a row reads that leave the priced program unchanged.
ROW_KEYS = ("inference_batch_K", "inference_rate_hz", "min_lifetime_hours")
#: [CL5] Declarations a priced platform may change against its scheduled program: the row keys and the physics.
PRICE_KEYS = ROW_KEYS + ("platform_physics_profile", "platform_physics_overrides", "platform_physics_resolved")


def require_declared_change(platform: Mapping[str, Any], reference: Mapping[str, Any], keys: Sequence[str],
                            what: str) -> None:
    """Refuse by name (``LadderDeclarationError``: "<what> only in <keys>") a ``platform`` that differs from
    ``reference`` in any declaration outside ``keys`` (``ROW_KEYS`` for a row, ``PRICE_KEYS`` for a pricing)."""
    if {k: v for k, v in platform.items() if k not in keys} != {k: v for k, v in reference.items() if k not in keys}:
        raise LadderDeclarationError(f"{what} only in " + ", ".join(keys))


def certificate_record(certificate: Any) -> Dict[str, Any]:
    """What every row prints of the network's conversion certificate (``certified``, ``summary()``)."""
    if certificate is None:
        return {"certified": False, "summary": "no conversion certificate was stamped on the flow"}
    return {"certified": bool(certificate.certified), "summary": str(certificate.summary())}


@dataclass(frozen=True)
class LadderNetwork:
    """A built network: its mapper representation and what the caller measured on its flow."""

    name: str
    mapper_repr: Any
    resolution: Tuple[int, ...]
    total_params: Optional[float] = None
    #: host_macs, onchip_macs, compute_op_count, host_params, onchip_params (absent: host share unknown).
    host_census: Optional[Mapping[str, int]] = None
    #: ``certificate_record`` of the conversion the mapper representation came from.
    conversion: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class ShapeCensus:
    """The shape-only layout of one network on one core geometry (independent of the core count)."""

    softcores: Tuple[Any, ...]
    wire_census: Any
    host_segments: int
    segments: int
    logical_cells: int
    as_mapped_cells: int
    wall_s: float
    #: [W03] The layout walk's program graph (``ProgramGraph.of_walk``), held with the wire census: a declared
    #: ``residency_plan`` is planned on it as the builder plans it.
    program: Any = None
    #: [fix_b] Per spec, its builder segment (``residency_rules.spec_runs``: the host ops before its node in the
    #: walk's program order); U1 plans these segments, as the builder does (``partition_ir_graph``).
    runs: Optional[Tuple[int, ...]] = None
    #: [U-E2 H-1] On a chip that runs its declared lanes (``scaling_ladder_lanes``): the host-op units the lanes
    #: absorb (they leave the host census), the lane work by kind as the IR fusion counts it, and the specs a declared
    #: ``residency_plan`` composes (segmented by the fused program's depth segments); 0 and None on any other chip.
    lane_absorbed: int = 0
    lane_ops: Optional[Mapping[str, int]] = None
    residency_softcores: Optional[Tuple[Any, ...]] = None
    #: [U-J] The walk's host-chip boundary (``mapping/support/host_boundary``): None without the wire census.
    host_boundary: Optional[Any] = None


@dataclass(frozen=True)
class ScheduledProgram:
    """One packed and scheduled program at one core budget, before pricing [CL5]: priced once per physics
    declaration and batch size (``price_scheduled``) without scheduling again."""

    platform: Mapping[str, Any]
    packed_tiles: int
    #: {"name": refusal class, "reason": ...} when the point does not schedule.
    refusal: Optional[Mapping[str, str]] = None
    stats: Any = None
    noc: Any = None
    facts: Any = None
    wall_s: float = 0.0


@dataclass(frozen=True)
class PricedProgram:
    """One packed program at one core budget: its pass census and four pricings."""

    platform: Mapping[str, Any]
    packed_tiles: int
    #: {"name": refusal class, "reason": ...} when the point does not schedule.
    refusal: Optional[Mapping[str, str]] = None
    passes: int = 0
    #: [fix_b] The period census's (pass, segment) rows: one per executed neural stage (an unscheduled pack is one
    #: planned pass of several segments, which the builder runs as one stage per segment).
    census_passes: Optional[int] = None
    latency_steps: Optional[int] = None
    reprogram_passes: int = 0
    #: Passes the period census rewrites every period (G3's ``write_census``, the executed rotation).
    written_passes: int = 0
    carry: Optional[Mapping[str, int]] = None
    #: The period census's rewrite bytes (``period_rewrite_bytes``), None where the census refuses.
    period_bytes: Optional[int] = None
    once_bytes: Optional[int] = None
    once_cores: int = 0
    max_writes: Tuple[Optional[float], str] = (None, "no pass census")
    endurance: Optional[float] = None
    terms: Mapping[str, Mapping[str, Band]] = field(default_factory=dict)
    refusals: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    wall_s: float = 0.0
    #: [CL5] The write law priced at the point's declared batch (``inference_batch_K``): its terms, refusals,
    #: admissibility verdicts and the headlines at that batch; empty when no batch is declared.
    write_law: Mapping[str, Any] = field(default_factory=dict)
    #: [CL5] The endurance band (low, nominal, high) where the physics declares one.
    endurance_band: Optional[Tuple[float, float, float]] = None
    #: [U-C C-1] The pulses per write (low, nominal, high) where the technology declares them.
    pulses_band: Optional[Tuple[float, float, float]] = None
    #: [U-C2 G-2] The physics' compute-time law (``cost.compute_time.compute_law``): ``t_cycle`` per timestep or
    #: ``t_synaptic_op`` per synaptic operation; None without physics.
    compute_law: Optional[str] = None
    #: [targeted M2] On several event-serial cores, the layer-pipelined stage floor (low, nominal, high):
    #: ``t_synaptic_op`` x the most loaded core's events of any pass (``cost.compute_time.busiest_stage``); None
    #: elsewhere.
    event_stage: Optional[Tuple[float, float, float]] = None
