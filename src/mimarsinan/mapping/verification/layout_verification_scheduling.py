from __future__ import annotations
from dataclasses import replace
from typing import Any, Optional, Sequence, Tuple
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.verification.layout_verification_packing import (
    _empty_stats, _stats_from_packing,
)
from mimarsinan.mapping.verification.layout_verification_types import LayoutVerificationStats
from mimarsinan.mapping.support.schedule.pass_planner import (
    plan_program_passes,
)
from mimarsinan.mapping.support.residency_rules import fit_rule_applies, passes_pinned, unscheduled_pack_holds
from mimarsinan.mapping.support.schedule.residency_composition import ShapeOnlyResidency, refuse_without_cut_census
from mimarsinan.mapping.packing.infeasibility_proofs import CoreCapacityRefusal, named_message


def compute_mapping_stats(
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    *,
    allow_scheduling: bool = False,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
    max_schedule_passes: int = 8,
    residency_plan: Optional[dict] = None,
    runs: Optional[Sequence[int]] = None,
    position_multiplexing: Optional[bool] = None,
    census: Optional[Any] = None,
    program: Optional[Any] = None,
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
    builder also consumes [U1]. [U1 W9] With the walk's ``runs``
    (``residency_rules.spec_runs``) it plans the builder's segments, so the
    searched program is the deployed one; without them it groups the specs by
    their depth segments, which is the deployed program only where both
    segmentations group the specs alike (every tier-0 template; a branchy graph
    whose host op splits one depth segment into two builder segments, or one
    builder segment holding two depth segments, differs on the base too).
    [W03] A declared ``residency_plan`` is planned with the layout walk's wire
    ``census`` and ``program`` graph (the residency composition the builder
    executes); a caller holding neither refuses it by name. [fix_b] Without
    ``runs``, the census's runs (``spec_runs``) group the specs where it states them. Where U1's passes do not
    fit the pool side by side (unpinned) while the unscheduled pack holds (``unscheduled_pack_holds``), that pack is
    the program (the fit rule, as the builder builds its single pool); [fix_c] likewise under a declared plan whose
    composition rotates tiles (r > 0), every plan but the stateless baseline (``fit_rule_applies``).
    """
    if census is None or program is None:
        refuse_without_cut_census("compute_mapping_stats", residency_plan)
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
        residency=None if residency_plan is None else ShapeOnlyResidency(residency_plan, census, program),
        runs=runs if runs is not None else getattr(census, "spec_runs", None),
        position_multiplexing=position_multiplexing,
    )

    if not plan.feasible:
        if flat_stats is not None:
            return flat_stats, None
        return _empty_stats(
            feasible=False, num_softcores=len(softcores),
            total_hw_cores=sum(int(ct.count) for ct in core_types),
        ), named_message(CoreCapacityRefusal(
            "Scheduling infeasible: at least one softcore cannot be packed "
            f"(segment(s) {list(plan.infeasible_segments())})"))

    # [R5-gate finding] A scheduled program's utilization is the AGGREGATE
    # over every pass (the record's crossbar sums per-pass allocations); the
    # old single-pass figure gated 33.3% against a measured 40.8% on the
    # deepcnn witness. Every pass is packed and the totals accumulate.
    best_stats = None
    committed = allocated_capacity = allocated_axons = allocated_neurons = 0
    used_axons_total = used_neurons_total = 0
    all_feasible = True
    packs = {}
    for index in sorted(range(len(plan.pass_lists)), key=lambda q: len(plan.pass_lists[q]), reverse=True):
        pr = packs[index] = pack_layout(
            softcores=plan.pass_lists[index],
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

    if fit_rule_applies(residency_plan) and flat_stats is not None and unscheduled_pack_holds(pack, core_types) and not (
            _pinned_whole(plan, [packs[q] for q in range(len(packs))], all_feasible, core_types)):
        return flat_stats, None  # [fix_b, fix_c] the fit rule: the planned passes write every period, the pack holds
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
        ), named_message(CoreCapacityRefusal("Scheduling: no pass could be packed"))

    return replace(
        best_stats,
        feasible=True,
        schedule_pass_count=plan.total_pass_count,
        schedule_sync_count=plan.sync_count,
        max_cores_per_pass=plan.budget,
    ), None


def _pinned_whole(plan: Any, packs: Sequence[Any], all_feasible: bool, core_types: Sequence[LayoutHardCoreType]) -> bool:
    """[fix_c] Whether the planned program keeps every core pinned: a declared plan's composition rotates no tile
    (r = 0), U1's passes fit the pool side by side (``passes_pinned``)."""
    if plan.residency is not None:
        return int(plan.residency.r) == 0
    return all_feasible and passes_pinned(list(packs), plan.resident_flags, core_types)
