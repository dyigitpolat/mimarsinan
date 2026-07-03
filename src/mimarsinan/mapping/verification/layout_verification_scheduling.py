from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.verification.layout_verification_packing import (
    _empty_stats, _stats_from_packing,
)
from mimarsinan.mapping.verification.layout_verification_types import LayoutVerificationStats
from mimarsinan.mapping.support.schedule.schedule_partitioner import (
    effective_core_budget,
    estimate_passes_for_layout_validated,
)


def compute_schedule_sync_count(per_segment_passes: Dict[int, int]) -> int:
    """Sync barriers from scheduled passes: a segment with N passes needs N-1 barriers."""
    return sum(max(n - 1, 0) for n in per_segment_passes.values())


def compute_mapping_stats(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    allow_scheduling: bool = False,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
) -> Tuple[LayoutVerificationStats, Optional[str]]:
    """Pack softcores and compute verification statistics with scheduling support.

    Returns ``(stats, None)`` on success (single-pass or scheduled) and
    ``(stats, error_message)`` when mapping is infeasible.
    """
    if not softcores or not core_types:
        return _empty_stats(feasible=False, num_softcores=len(softcores)), \
            "No softcores or core types"

    pack = pack_layout(
        softcores=softcores,
        core_types=core_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_coalescing=allow_coalescing,
    )

    if pack.feasible:
        return _stats_from_packing(
            pack, num_original_softcores=len(softcores),
            softcores=softcores, core_types=core_types,
        ), None

    if not allow_scheduling:
        return _empty_stats(
            feasible=False, num_softcores=len(softcores),
            total_hw_cores=sum(int(ct.count) for ct in core_types),
        ), pack.error or "HW bin-packing infeasible"

    core_dicts = [
        {"max_axons": ct.max_axons, "max_neurons": ct.max_neurons, "count": ct.count}
        for ct in core_types
    ]
    budget = effective_core_budget(core_dicts)
    max_hw_ax = max(ct.max_axons for ct in core_types)
    max_hw_neu = max(ct.max_neurons for ct in core_types)

    seg_softcores: Dict[int, List[LayoutSoftCoreSpec]] = {}
    for sc in softcores:
        sid = sc.segment_id if sc.segment_id is not None else 0
        seg_softcores.setdefault(sid, []).append(sc)

    per_segment_passes: Dict[int, int] = {}
    total_pass_count = 0
    all_pass_lists: List[List[LayoutSoftCoreSpec]] = []
    sched_feasible = True

    common_kwargs = dict(
        max_hw_axons=max_hw_ax,
        max_hw_neurons=max_hw_neu,
        allow_coalescing=allow_coalescing,
        allow_splitting=allow_neuron_splitting,
        core_types=core_dicts,
    )

    for sid in sorted(seg_softcores.keys()):
        seg_scs = seg_softcores[sid]
        if budget > 0:
            n_passes, seg_pass_lists, seg_ok = estimate_passes_for_layout_validated(
                seg_scs, budget, **common_kwargs,
            )
            if not seg_ok:
                sched_feasible = False
        else:
            n_passes, seg_pass_lists = 1, [seg_scs]
        per_segment_passes[sid] = max(n_passes, 1)
        total_pass_count += max(n_passes, 1)
        all_pass_lists.extend(seg_pass_lists)

    if not sched_feasible:
        return _empty_stats(
            feasible=False, num_softcores=len(softcores),
            total_hw_cores=sum(int(ct.count) for ct in core_types),
        ), "Scheduling infeasible: at least one softcore cannot be packed"

    best_stats = None
    for pass_scs in sorted(all_pass_lists, key=len, reverse=True):
        try:
            pr = pack_layout(
                softcores=pass_scs,
                core_types=core_types,
                allow_neuron_splitting=allow_neuron_splitting,
                allow_coalescing=allow_coalescing,
            )
            if pr.feasible:
                best_stats = _stats_from_packing(
                    pr, num_original_softcores=len(softcores),
                    softcores=softcores, core_types=core_types,
                )
                break
        except Exception:
            continue

    if best_stats is None:
        return _empty_stats(
            feasible=False, num_softcores=len(softcores),
            total_hw_cores=sum(int(ct.count) for ct in core_types),
        ), "Scheduling: no pass could be packed"

    sync_count = compute_schedule_sync_count(per_segment_passes)
    return replace(
        best_stats,
        feasible=True,
        schedule_pass_count=total_pass_count,
        schedule_sync_count=sync_count,
        max_cores_per_pass=budget,
    ), None
