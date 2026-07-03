from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class LayoutVerificationStats:
    """Hardware-mapping performance metrics derived from layout packing."""

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

    neural_segment_count: int = 0
    segment_latency_min: float = 0.0
    segment_latency_median: float = 0.0
    segment_latency_max: float = 0.0
    threshold_group_count: int = 0

    coalescing_group_count: int = 0
    coalescing_frags_per_group_min: float = 0.0
    coalescing_frags_per_group_median: float = 0.0
    coalescing_frags_per_group_max: float = 0.0

    split_softcore_count: int = 0
    splits_per_softcore_min: float = 0.0
    splits_per_softcore_median: float = 0.0
    splits_per_softcore_max: float = 0.0

    schedule_pass_count: int = 0
    schedule_sync_count: int = 0
    max_cores_per_pass: int = 0

    unused_area_total: int = 0
    unusable_space_total: int = 0
    fragmentation_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

