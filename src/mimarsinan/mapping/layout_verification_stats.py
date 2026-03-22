"""Reusable layout-verification statistics.

Pure function that computes hardware-mapping performance metrics from
layout-packing results.  UI-agnostic: the same ``LayoutVerificationStats``
can be consumed by the wizard, monitor, search, or reporting code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import (
    LayoutCoreSnapshot,
    LayoutHardCoreType,
    LayoutPackingResult,
    LayoutSoftCoreSpec,
)


@dataclass(frozen=True)
class LayoutVerificationStats:
    """Hardware-mapping performance metrics derived from layout packing."""

    feasible: bool

    total_cores: int
    total_softcores: int

    # Total wasted axons/neurons as a percentage of total available on used cores.
    total_wasted_axons_pct: float
    total_wasted_neurons_pct: float

    # Mapped parameters (area) as a percentage of total hardware capacity.
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
    max_cores_per_pass: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pct(part: float, total: float) -> float:
    return (part / total * 100.0) if total > 0 else 0.0


def _safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return float(s[n // 2]) if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _latency_stats(
    softcores: Optional[Sequence[LayoutSoftCoreSpec]],
) -> tuple[int, float, float, float, int]:
    """Return segment count, latency tiers/segment min-median-max, threshold count."""
    if not softcores:
        return 0, 0.0, 0.0, 0.0, 0

    tagged_softcores = [sc for sc in softcores if sc.latency_tag is not None]
    threshold_groups = len({sc.threshold_group_id for sc in softcores})
    if not tagged_softcores:
        return 0, 0.0, 0.0, 0.0, threshold_groups

    segments_to_latencies: Dict[int, set[int]] = {}
    fallback_by_latency_tag = {int(sc.latency_tag) for sc in tagged_softcores}
    has_segment_ids = any(sc.segment_id is not None for sc in tagged_softcores)

    if has_segment_ids:
        for sc in tagged_softcores:
            seg_id = int(sc.segment_id) if sc.segment_id is not None else int(sc.latency_tag)
            segments_to_latencies.setdefault(seg_id, set()).add(int(sc.latency_tag))
    else:
        # Backward-compatible fallback for handcrafted tests / old softcore data:
        # each distinct latency tag is treated as its own single-tier segment.
        for lat in fallback_by_latency_tag:
            segments_to_latencies[lat] = {lat}

    per_segment_latencies = sorted(float(len(latencies)) for latencies in segments_to_latencies.values())
    return (
        len(per_segment_latencies),
        float(min(per_segment_latencies)),
        _safe_median(per_segment_latencies),
        float(max(per_segment_latencies)),
        threshold_groups,
    )


def _stats_from_packing(
    packing: LayoutPackingResult,
    num_original_softcores: int,
    softcores: Optional[Sequence[LayoutSoftCoreSpec]] = None,
) -> LayoutVerificationStats:
    """Build stats from a successful packing result."""
    snaps = packing.used_core_snapshots or ()

    if not snaps:
        return _empty_stats(feasible=packing.feasible, num_softcores=num_original_softcores)

    total_axons = sum(s.axons_per_core for s in snaps)
    total_neurons = sum(s.neurons_per_core for s in snaps)
    total_wasted_ax = sum(s.wasted_axons for s in snaps)
    total_wasted_neu = sum(s.wasted_neurons for s in snaps)

    per_core_ax_pct = [_pct(s.wasted_axons, s.axons_per_core) for s in snaps]
    per_core_neu_pct = [_pct(s.wasted_neurons, s.neurons_per_core) for s in snaps]
    per_core_param_pct = [_pct(s.used_area, s.capacity) for s in snaps]

    # Neural segment / latency summary and threshold groups from the original
    # softcores list.
    (
        neural_segment_count,
        segment_latency_min,
        segment_latency_median,
        segment_latency_max,
        threshold_group_count,
    ) = _latency_stats(softcores)

    # Coalescing group distribution.
    coal_sizes = list(packing.coalescing_group_sizes or ())
    coalescing_group_count = len(coal_sizes)
    coal_sizes_f = [float(x) for x in coal_sizes]
    coalescing_frags_per_group_min = float(min(coal_sizes_f)) if coal_sizes_f else 0.0
    coalescing_frags_per_group_median = _safe_median(coal_sizes_f)
    coalescing_frags_per_group_max = float(max(coal_sizes_f)) if coal_sizes_f else 0.0

    # Split distribution.
    split_counts = list(packing.split_counts_per_sc or ())
    split_softcore_count = len(split_counts)
    split_counts_f = [float(x) for x in split_counts]
    splits_per_softcore_min = float(min(split_counts_f)) if split_counts_f else 0.0
    splits_per_softcore_median = _safe_median(split_counts_f)
    splits_per_softcore_max = float(max(split_counts_f)) if split_counts_f else 0.0

    return LayoutVerificationStats(
        feasible=packing.feasible,
        total_cores=packing.cores_used,
        total_softcores=num_original_softcores,
        total_wasted_axons_pct=_pct(total_wasted_ax, total_axons),
        total_wasted_neurons_pct=_pct(total_wasted_neu, total_neurons),
        mapped_params_pct=_pct(packing.used_area, packing.total_capacity),
        per_core_wasted_axons_pct_min=min(per_core_ax_pct),
        per_core_wasted_axons_pct_avg=sum(per_core_ax_pct) / len(per_core_ax_pct),
        per_core_wasted_axons_pct_max=max(per_core_ax_pct),
        per_core_wasted_neurons_pct_min=min(per_core_neu_pct),
        per_core_wasted_neurons_pct_avg=sum(per_core_neu_pct) / len(per_core_neu_pct),
        per_core_wasted_neurons_pct_max=max(per_core_neu_pct),
        per_core_mapped_params_pct_min=min(per_core_param_pct),
        per_core_mapped_params_pct_avg=sum(per_core_param_pct) / len(per_core_param_pct),
        per_core_mapped_params_pct_max=max(per_core_param_pct),
        coalesced_cores=packing.coalesced_fragment_count,
        split_cores=packing.split_fragment_count,
        neural_segment_count=neural_segment_count,
        segment_latency_min=segment_latency_min,
        segment_latency_median=segment_latency_median,
        segment_latency_max=segment_latency_max,
        threshold_group_count=threshold_group_count,
        coalescing_group_count=coalescing_group_count,
        coalescing_frags_per_group_min=coalescing_frags_per_group_min,
        coalescing_frags_per_group_median=coalescing_frags_per_group_median,
        coalescing_frags_per_group_max=coalescing_frags_per_group_max,
        split_softcore_count=split_softcore_count,
        splits_per_softcore_min=splits_per_softcore_min,
        splits_per_softcore_median=splits_per_softcore_median,
        splits_per_softcore_max=splits_per_softcore_max,
    )


def _empty_stats(*, feasible: bool, num_softcores: int = 0) -> LayoutVerificationStats:
    return LayoutVerificationStats(
        feasible=feasible,
        total_cores=0,
        total_softcores=num_softcores,
        total_wasted_axons_pct=0.0,
        total_wasted_neurons_pct=0.0,
        mapped_params_pct=0.0,
        per_core_wasted_axons_pct_min=0.0,
        per_core_wasted_axons_pct_avg=0.0,
        per_core_wasted_axons_pct_max=0.0,
        per_core_wasted_neurons_pct_min=0.0,
        per_core_wasted_neurons_pct_avg=0.0,
        per_core_wasted_neurons_pct_max=0.0,
        per_core_mapped_params_pct_min=0.0,
        per_core_mapped_params_pct_avg=0.0,
        per_core_mapped_params_pct_max=0.0,
        coalesced_cores=0,
        split_cores=0,
        neural_segment_count=0,
        segment_latency_min=0.0,
        segment_latency_median=0.0,
        segment_latency_max=0.0,
        threshold_group_count=0,
        coalescing_group_count=0,
        coalescing_frags_per_group_min=0.0,
        coalescing_frags_per_group_median=0.0,
        coalescing_frags_per_group_max=0.0,
        split_softcore_count=0,
        splits_per_softcore_min=0.0,
        splits_per_softcore_median=0.0,
        splits_per_softcore_max=0.0,
    )


def build_layout_verification_stats(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    allow_neuron_splitting: bool = False,
    allow_axon_coalescing: bool = False,
) -> LayoutVerificationStats:
    """Pack softcores and compute verification statistics.

    This is the primary public entry point.  It runs ``pack_layout`` internally
    and derives all metrics from the packing result.
    """
    if not softcores or not core_types:
        return _empty_stats(feasible=False, num_softcores=len(softcores))

    packing = pack_layout(
        softcores=softcores,
        core_types=core_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_axon_coalescing=allow_axon_coalescing,
    )

    if not packing.feasible:
        return _empty_stats(feasible=False, num_softcores=len(softcores))

    return _stats_from_packing(packing, num_original_softcores=len(softcores), softcores=softcores)


def build_stats_from_packing_result(
    packing: LayoutPackingResult,
    num_original_softcores: int,
    softcores: Optional[Sequence[LayoutSoftCoreSpec]] = None,
) -> LayoutVerificationStats:
    """Build stats from an already-computed packing result.

    Use when the caller already has a ``LayoutPackingResult`` (e.g. from
    ``verify_hardware_config``).  Pass ``softcores`` to include latency and
    threshold-group metrics.
    """
    if not packing.feasible:
        return _empty_stats(feasible=False, num_softcores=num_original_softcores)
    return _stats_from_packing(packing, num_original_softcores, softcores=softcores)
