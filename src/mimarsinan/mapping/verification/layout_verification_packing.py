from __future__ import annotations
from typing import Optional, Sequence
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType, LayoutPackingResult, LayoutSoftCoreSpec,
)
from mimarsinan.mapping.verification.layout_verification_helpers import (
    _latency_stats, _pct, _safe_median,
)
from mimarsinan.mapping.verification.layout_verification_types import LayoutVerificationStats
def _stats_from_packing(
    packing: LayoutPackingResult,
    num_original_softcores: int,
    softcores: Optional[Sequence[LayoutSoftCoreSpec]] = None,
    core_types: Optional[Sequence[LayoutHardCoreType]] = None,
) -> LayoutVerificationStats:
    """Build stats from a successful packing result.

    [H3] Percentage metrics are computed over the ALLOCATED cores only — the
    estimand the sealed record measures (`CrossbarUtilizationReport`), so the
    candidate's utilization/wastage figures zip against the deployment's
    instead of forking on the denominator (measured 2.00x on the study MLP).
    The whole-chip signal keeps its own name: ``chip_occupancy_pct`` divides
    by the DECLARED chip (idle cores included), which is what a search that
    wants chip-shrinking pressure selects. The numerator is the record's too:
    the committed RECTANGLE (used rows x used columns per core — IMC reclaims
    whole rows/columns only), not the sum of placed pieces, which undercounts
    the structure a diagonal packing strands.
    """
    snaps = packing.used_core_snapshots or ()

    if not snaps:
        return _empty_stats(feasible=packing.feasible, num_softcores=num_original_softcores)

    if core_types:
        chip_total_cores = sum(int(ct.count) for ct in core_types)
        chip_total_axons = sum(int(ct.max_axons) * int(ct.count) for ct in core_types)
        chip_total_neurons = sum(int(ct.max_neurons) * int(ct.count) for ct in core_types)
        chip_total_capacity = sum(
            int(ct.max_axons) * int(ct.max_neurons) * int(ct.count)
            for ct in core_types
        )
    else:
        chip_total_cores = 0
        chip_total_axons = sum(s.axons_per_core for s in snaps)
        chip_total_neurons = sum(s.neurons_per_core for s in snaps)
        chip_total_capacity = packing.total_capacity

    total_used_axons = sum(s.used_axons for s in snaps)
    total_used_neurons = sum(s.used_neurons for s in snaps)
    allocated_axons = sum(s.axons_per_core for s in snaps)
    allocated_neurons = sum(s.neurons_per_core for s in snaps)
    allocated_capacity = sum(s.capacity for s in snaps)
    committed_cells = sum(s.used_axons * s.used_neurons for s in snaps)

    # Per-core spreads are ALLOCATED-only too: an idle core is not a badly
    # packed core, it is chip headroom — chip_occupancy_pct's business.
    per_core_ax_pct = [_pct(s.wasted_axons, s.axons_per_core) for s in snaps]
    per_core_neu_pct = [_pct(s.wasted_neurons, s.neurons_per_core) for s in snaps]
    per_core_param_pct = [
        _pct(s.used_axons * s.used_neurons, s.capacity) for s in snaps
    ]

    (
        neural_segment_count,
        segment_latency_min,
        segment_latency_median,
        segment_latency_max,
        residency_class_count,
    ) = _latency_stats(softcores)

    coal_sizes = list(packing.coalescing_group_sizes or ())
    coalescing_group_count = len(coal_sizes)
    coal_sizes_f = [float(x) for x in coal_sizes]
    coalescing_frags_per_group_min = float(min(coal_sizes_f)) if coal_sizes_f else 0.0
    coalescing_frags_per_group_median = _safe_median(coal_sizes_f)
    coalescing_frags_per_group_max = float(max(coal_sizes_f)) if coal_sizes_f else 0.0

    split_counts = list(packing.split_counts_per_sc or ())
    split_softcore_count = len(split_counts)
    split_counts_f = [float(x) for x in split_counts]
    splits_per_softcore_min = float(min(split_counts_f)) if split_counts_f else 0.0
    splits_per_softcore_median = _safe_median(split_counts_f)
    splits_per_softcore_max = float(max(split_counts_f)) if split_counts_f else 0.0

    frag_cap = int(packing.total_capacity)
    fragmentation_pct = _pct(float(packing.unusable_space_total), float(frag_cap))

    return LayoutVerificationStats(
        feasible=packing.feasible,
        total_cores=packing.cores_used,
        total_hw_cores=chip_total_cores,
        total_softcores=num_original_softcores,
        total_wasted_axons_pct=_pct(
            allocated_axons - total_used_axons, allocated_axons),
        total_wasted_neurons_pct=_pct(
            allocated_neurons - total_used_neurons, allocated_neurons),
        mapped_params_pct=_pct(committed_cells, allocated_capacity),
        chip_occupancy_pct=_pct(committed_cells, chip_total_capacity),
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
        residency_class_count=residency_class_count,
        coalescing_group_count=coalescing_group_count,
        coalescing_frags_per_group_min=coalescing_frags_per_group_min,
        coalescing_frags_per_group_median=coalescing_frags_per_group_median,
        coalescing_frags_per_group_max=coalescing_frags_per_group_max,
        split_softcore_count=split_softcore_count,
        splits_per_softcore_min=splits_per_softcore_min,
        splits_per_softcore_median=splits_per_softcore_median,
        splits_per_softcore_max=splits_per_softcore_max,
        unused_area_total=int(packing.unused_area_total),
        unusable_space_total=int(packing.unusable_space_total),
        fragmentation_pct=fragmentation_pct,
    )


def _empty_stats(*, feasible: bool, num_softcores: int = 0, total_hw_cores: int = 0) -> LayoutVerificationStats:
    return LayoutVerificationStats(
        feasible=feasible,
        total_cores=0,
        total_hw_cores=total_hw_cores,
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
        residency_class_count=0,
        coalescing_group_count=0,
        coalescing_frags_per_group_min=0.0,
        coalescing_frags_per_group_median=0.0,
        coalescing_frags_per_group_max=0.0,
        split_softcore_count=0,
        splits_per_softcore_min=0.0,
        splits_per_softcore_median=0.0,
        splits_per_softcore_max=0.0,
        unused_area_total=0,
        unusable_space_total=0,
        fragmentation_pct=0.0,
    )


def build_layout_verification_stats(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
) -> LayoutVerificationStats:
    """Pack softcores via ``pack_layout`` and derive all verification metrics (primary public entry point)."""
    if not softcores or not core_types:
        return _empty_stats(feasible=False, num_softcores=len(softcores))

    packing = pack_layout(
        softcores=softcores,
        core_types=core_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_coalescing=allow_coalescing,
    )

    if not packing.feasible:
        return _empty_stats(feasible=False, num_softcores=len(softcores))

    return _stats_from_packing(
        packing,
        num_original_softcores=len(softcores),
        softcores=softcores,
        core_types=core_types,
    )


def build_stats_from_packing_result(
    packing: LayoutPackingResult,
    num_original_softcores: int,
    softcores: Optional[Sequence[LayoutSoftCoreSpec]] = None,
    core_types: Optional[Sequence[LayoutHardCoreType]] = None,
) -> LayoutVerificationStats:
    """Build stats from an already-computed packing result.

    Pass ``softcores`` to include latency/residency-class metrics; pass
    ``core_types`` to compute metrics against the full chip (including idle cores).
    """
    if not packing.feasible:
        total_hw = sum(int(ct.count) for ct in core_types) if core_types else 0
        return _empty_stats(feasible=False, num_softcores=num_original_softcores, total_hw_cores=total_hw)
    return _stats_from_packing(
        packing, num_original_softcores, softcores=softcores, core_types=core_types,
    )
