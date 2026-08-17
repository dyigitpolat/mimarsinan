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
)
from mimarsinan.mapping.support.schedule.schedule_policy import (
    plan_segment_passes,
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
    schedule_policy: str = "pool",
    max_schedule_passes: int = 8,
) -> Tuple[LayoutVerificationStats, Optional[str]]:
    """Pack softcores and compute verification statistics with scheduling support.

    Returns ``(stats, None)`` on success (single-pass or scheduled) and
    ``(stats, error_message)`` when mapping is infeasible.

    ``schedule_policy``/``max_schedule_passes`` are the SAME declared knobs the
    hard-core builder consumes. With ``allow_scheduling`` on, the pass structure is
    composed even when the flat pack would fit, because the builder composes it too
    (``allow_scheduling`` always routes through the scheduled build): a scheduled
    platform is otherwise searched against a program it will never run. That holds
    whatever the POLICY — the policy decides WHICH schedule is composed
    (bank-clustered residency, else the capacity split), never whether one exists —
    so a fitting pool platform reports the passes it will actually run rather than
    the zero the flat pack would suggest [C4].
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
    flat_stats = (
        _stats_from_packing(
            pack, num_original_softcores=len(softcores),
            softcores=softcores, core_types=core_types,
        )
        if pack.feasible
        else None
    )
    # A scheduled deployment runs passes even when everything would fit at once.
    composes_over_a_fitting_pack = allow_scheduling

    if flat_stats is not None and not composes_over_a_fitting_pack:
        return flat_stats, None

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

    seg_softcores: Dict[int, List[LayoutSoftCoreSpec]] = {}
    for sc in softcores:
        sid = sc.segment_id if sc.segment_id is not None else 0
        seg_softcores.setdefault(sid, []).append(sc)

    per_segment_passes: Dict[int, int] = {}
    total_pass_count = 0
    all_pass_lists: List[List[LayoutSoftCoreSpec]] = []
    sched_feasible = True
    policy_applied = False

    for sid in sorted(seg_softcores.keys()):
        n_passes, seg_pass_lists, seg_ok, seg_policy = plan_segment_passes(
            seg_softcores[sid], budget,
            core_types=core_types,
            allow_coalescing=allow_coalescing,
            allow_splitting=allow_neuron_splitting,
            schedule_policy=schedule_policy,
            max_schedule_passes=max_schedule_passes,
        )
        if not seg_ok:
            sched_feasible = False
        policy_applied = policy_applied or seg_policy
        per_segment_passes[sid] = max(n_passes, 1)
        total_pass_count += max(n_passes, 1)
        all_pass_lists.extend(seg_pass_lists)

    if not sched_feasible:
        if flat_stats is not None:
            return flat_stats, None
        return _empty_stats(
            feasible=False, num_softcores=len(softcores),
            total_hw_cores=sum(int(ct.count) for ct in core_types),
        ), "Scheduling infeasible: at least one softcore cannot be packed"

    # [R5-gate finding] A scheduled program's utilization is the AGGREGATE
    # over every pass (the record's crossbar sums per-pass allocations); the
    # old single-pass figure gated 33.3% against a measured 40.8% on the
    # deepcnn witness. Every pass is packed and the totals accumulate.
    best_stats = None
    committed = allocated_capacity = allocated_axons = allocated_neurons = 0
    used_axons_total = used_neurons_total = 0
    all_feasible = True
    for pass_scs in sorted(all_pass_lists, key=len, reverse=True):
        pr = pack_layout(
            softcores=pass_scs,
            core_types=core_types,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )
        if not pr.feasible:
            all_feasible = False
            continue
        for snap in pr.used_core_snapshots or ():
            committed += snap.used_axons * snap.used_neurons
            allocated_capacity += snap.capacity
            allocated_axons += snap.axons_per_core
            allocated_neurons += snap.neurons_per_core
            used_axons_total += snap.used_axons
            used_neurons_total += snap.used_neurons
        if best_stats is None:
            best_stats = _stats_from_packing(
                pr, num_original_softcores=len(softcores),
                softcores=softcores, core_types=core_types,
            )

    if best_stats is not None and all_feasible and allocated_capacity > 0:
        best_stats = replace(
            best_stats,
            mapped_params_pct=100.0 * committed / allocated_capacity,
            total_wasted_axons_pct=100.0
            * (allocated_axons - used_axons_total) / max(allocated_axons, 1),
            total_wasted_neurons_pct=100.0
            * (allocated_neurons - used_neurons_total)
            / max(allocated_neurons, 1),
        )

    if best_stats is None:
        if flat_stats is not None:
            return flat_stats, None
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
