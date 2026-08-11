"""Spike traffic: per boundary (gate-reduced) and per NoC link (SANA-FE scope)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.deployment_record.schema.serde import optional, strict_kwargs, tuple_of


@dataclass(frozen=True)
class BoundaryTrafficRecord:
    """Per-boundary spike totals, REDUCED at the gate — raw tensors never persist."""

    node_id: int
    producing_stage_index: Optional[int]
    neurons: int
    samples: int
    total_count: int
    max_neuron_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundaryTrafficRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class NocLinkLoadRecord:
    """Packet count over one directed mesh link."""

    from_x: int
    from_y: int
    to_x: int
    to_y: int
    packet_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NocLinkLoadRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class NocTrafficRecord:
    """SANA-FE NoC totals, summed over segments and samples."""

    total_packets: int
    inter_tile_packets: int
    intra_tile_packets: int
    input_path_packets: int
    cross_tile_connectivity_edges: int
    mapped_cross_tile_axons: int
    link_loads: Tuple[NocLinkLoadRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NocTrafficRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["link_loads"] = tuple_of(NocLinkLoadRecord.from_dict, kwargs["link_loads"])
        return cls(**kwargs)


@dataclass(frozen=True)
class TrafficRecord:
    """The traffic fragment: boundaries when counts-observable + gate armed; noc with SANA-FE."""

    boundaries: Optional[Tuple[BoundaryTrafficRecord, ...]]
    noc: Optional[NocTrafficRecord]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrafficRecord":
        kwargs = strict_kwargs(cls, data)
        boundaries = kwargs["boundaries"]
        kwargs["boundaries"] = (
            None
            if boundaries is None
            else tuple_of(BoundaryTrafficRecord.from_dict, boundaries)
        )
        kwargs["noc"] = optional(NocTrafficRecord.from_dict, kwargs["noc"])
        return cls(**kwargs)
