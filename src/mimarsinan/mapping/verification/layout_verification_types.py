from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class LayoutVerificationStats:
    """Hardware-mapping performance metrics derived from layout packing."""

    feasible: bool

    total_cores: int
    total_softcores: int

    # Total hardware cores on the chip (used + idle).  When chip-level
    # accounting is active this equals sum(ct.count); otherwise 0.
    total_hw_cores: int

    # Total wasted axons/neurons as a percentage of total available across
    # the entire chip (including idle cores whose resources are fully wasted).
    total_wasted_axons_pct: float
    total_wasted_neurons_pct: float

    # Mapped parameters (area) as a percentage of total chip capacity.
    mapped_params_pct: float

    # Per-core min/avg/max wasted axons percentage.
    per_core_wasted_axons_pct_min: float
    per_core_wasted_axons_pct_avg: float
    per_core_wasted_axons_pct_max: float

    # Per-core min/avg/max wasted neurons percentage.
    per_core_wasted_neurons_pct_min: float
    per_core_wasted_neurons_pct_avg: float
    per_core_wasted_neurons_pct_max: float

    # Per-core min/avg/max mapped-parameter utilization percentage.
    per_core_mapped_params_pct_min: float
    per_core_mapped_params_pct_avg: float
    per_core_mapped_params_pct_max: float

    # Coalescing and neuron-splitting fragment counts.
    coalesced_cores: int
    split_cores: int

    # Neural segment / latency summary metrics (derived from the original
    # softcores list). ``segment_latency_*`` summarizes the number of distinct
    # latency tiers inside each neural segment, not the global latency-tag values.
    neural_segment_count: int = 0
    segment_latency_min: float = 0.0
    segment_latency_median: float = 0.0
    segment_latency_max: float = 0.0
    threshold_group_count: int = 0

    # Coalescing group distribution (one entry per group that produced > 1 fragment).
    coalescing_group_count: int = 0
    coalescing_frags_per_group_min: float = 0.0
    coalescing_frags_per_group_median: float = 0.0
    coalescing_frags_per_group_max: float = 0.0

    # Split distribution (one entry per softcore that was split ≥ once).
    split_softcore_count: int = 0
    splits_per_softcore_min: float = 0.0
    splits_per_softcore_median: float = 0.0
    splits_per_softcore_max: float = 0.0

    # Scheduled mapping metrics.
    schedule_pass_count: int = 0
    schedule_sync_count: int = 0
    max_cores_per_pass: int = 0

    # Rectangular leftover on used cores (from packing); strip-shaped internal
    # fragmentation (unusable_space) vs ``unused_area_total`` (orthogonal signals).
    unused_area_total: int = 0
    unusable_space_total: int = 0
    fragmentation_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

