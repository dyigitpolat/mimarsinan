from __future__ import annotations
from dataclasses import replace
from typing import Optional, Sequence, Tuple
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.verification.layout_verification_packing import (
    _empty_stats, _stats_from_packing,
)
from mimarsinan.mapping.verification.layout_verification_types import LayoutVerificationStats
from mimarsinan.mapping.support.schedule.pass_planner import (
    plan_program_passes,
)


def compute_mapping_stats(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    allow_scheduling: bool = False,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
    max_schedule_passes: int = 8,
) -> Tuple[LayoutVerificationStats, Optional[str]]:
    """Pack softcores and compute verification statistics with scheduling support.

    Returns ``(stats, None)`` on success (single-pass or scheduled) and
    ``(stats, error_message)`` when mapping is infeasible.

    With ``allow_scheduling`` on, the pass structure is composed even when the
    flat pack would fit, because the builder composes it too
    (``allow_scheduling`` always routes through the scheduled build): a
    scheduled platform is otherwise searched against a program it will never
    run [C4]. The composition itself comes from the ONE planner
    (``pass_planner``: residency-first, capacity fallback) that the hard-core
    builder also consumes [U1] — so the searched program is the deployed one
    by construction, not by parity pinning.
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

    if flat_stats is not None and not allow_scheduling:
        return flat_stats, None

    if not allow_scheduling:
        return _empty_stats(
            feasible=False, num_softcores=len(softcores),
            total_hw_cores=sum(int(ct.count) for ct in core_types),
        ), pack.error or "HW bin-packing infeasible"

    plan = plan_program_passes(
        softcores, core_types,
        allow_coalescing=allow_coalescing,
        allow_splitting=allow_neuron_splitting,
        max_schedule_passes=max_schedule_passes,
    )

    if not plan.feasible:
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
    for pass_scs in sorted(plan.pass_lists, key=len, reverse=True):
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

    return replace(
        best_stats,
        feasible=True,
        schedule_pass_count=plan.total_pass_count,
        schedule_sync_count=plan.sync_count,
        max_cores_per_pass=plan.budget,
    ), None
