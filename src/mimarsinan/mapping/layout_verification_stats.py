"""Reusable layout-verification statistics.

Pure function that computes hardware-mapping performance metrics from
layout-packing results.  UI-agnostic: the same ``LayoutVerificationStats``
can be consumed by the wizard, monitor, search, or reporting code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pct(part: float, total: float) -> float:
    return (part / total * 100.0) if total > 0 else 0.0


def _stats_from_packing(
    packing: LayoutPackingResult,
    num_original_softcores: int,
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

    return _stats_from_packing(packing, num_original_softcores=len(softcores))


def build_stats_from_packing_result(
    packing: LayoutPackingResult,
    num_original_softcores: int,
) -> LayoutVerificationStats:
    """Build stats from an already-computed packing result.

    Use when the caller already has a ``LayoutPackingResult`` (e.g. from
    ``verify_hardware_config``).
    """
    if not packing.feasible:
        return _empty_stats(feasible=False, num_softcores=num_original_softcores)
    return _stats_from_packing(packing, num_original_softcores)
