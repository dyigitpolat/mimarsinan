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
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.schema.provenance import Band


class LadderGeometryError(ToolchainRefusal, ValueError):
    """The census needs an on-chip softcore; the packed tile count needs one declared core shape and a
    program that packs on unbounded cores."""


class LadderDeclarationError(ToolchainRefusal, ValueError):
    """A row's platform differs from its priced program in more than K, rate and lifetime."""


class LadderScheduleRefusal(ToolchainRefusal, ValueError):
    """The builder's planner finds no schedule at the point's core budget (a row records it by name)."""


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


@dataclass(frozen=True)
class PricedProgram:
    """One packed program at one core budget: its pass census and four pricings."""

    platform: Mapping[str, Any]
    packed_tiles: int
    #: {"name": refusal class, "reason": ...} when the point does not schedule.
    refusal: Optional[Mapping[str, str]] = None
    passes: int = 0
    latency_steps: Optional[int] = None
    reprogram_passes: int = 0
    #: Passes that place cores on non-resident weights (``law.WRITES_RULE`` reads it).
    written_passes: int = 0
    carry: Optional[Mapping[str, int]] = None
    period_bytes: Optional[int] = None
    once_bytes: Optional[int] = None
    once_cores: int = 0
    max_writes: Tuple[Optional[float], str] = (None, "no pass census")
    endurance: Optional[float] = None
    terms: Mapping[str, Mapping[str, Band]] = field(default_factory=dict)
    refusals: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    wall_s: float = 0.0
