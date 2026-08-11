"""Utilization/area: the two formerly write-only mapping reports, folded and typed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from mimarsinan.deployment_record.schema.serde import strict_kwargs


@dataclass(frozen=True)
class CrossbarUtilizationRecord:
    """Typed mirror of ``CrossbarUtilizationReport.to_dict()`` — same 14 keys."""

    cores_allocated: int
    axons_used: int
    axons_physical: int
    axon_utilization: float
    neurons_used: int
    neurons_physical: int
    neuron_utilization: float
    cells_used: int
    cells_physical: int
    cell_occupancy: float
    unusable_space: int
    macs: int
    weight_bits: Optional[int]
    programming_bits: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CrossbarUtilizationRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class LayoutStatsRecord:
    """Typed mirror of ``LayoutVerificationStats`` — every field, same names."""

    feasible: bool
    total_cores: int
    total_softcores: int
    total_hw_cores: int
    total_wasted_axons_pct: float
    total_wasted_neurons_pct: float
    mapped_params_pct: float
    per_core_wasted_axons_pct_min: float
    per_core_wasted_axons_pct_avg: float
    per_core_wasted_axons_pct_max: float
    per_core_wasted_neurons_pct_min: float
    per_core_wasted_neurons_pct_avg: float
    per_core_wasted_neurons_pct_max: float
    per_core_mapped_params_pct_min: float
    per_core_mapped_params_pct_avg: float
    per_core_mapped_params_pct_max: float
    coalesced_cores: int
    split_cores: int
    neural_segment_count: int
    segment_latency_min: float
    segment_latency_median: float
    segment_latency_max: float
    residency_class_count: int
    coalescing_group_count: int
    coalescing_frags_per_group_min: float
    coalescing_frags_per_group_median: float
    coalescing_frags_per_group_max: float
    split_softcore_count: int
    splits_per_softcore_min: float
    splits_per_softcore_median: float
    splits_per_softcore_max: float
    schedule_pass_count: int
    schedule_sync_count: int
    max_cores_per_pass: int
    unused_area_total: int
    unusable_space_total: int
    fragmentation_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayoutStatsRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class UtilizationRecord:
    """The utilization fragment, plus the formerly-discarded relay-core count."""

    crossbar: CrossbarUtilizationRecord
    layout: LayoutStatsRecord
    relay_cores_inserted: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UtilizationRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["crossbar"] = CrossbarUtilizationRecord.from_dict(kwargs["crossbar"])
        kwargs["layout"] = LayoutStatsRecord.from_dict(kwargs["layout"])
        return cls(**kwargs)
