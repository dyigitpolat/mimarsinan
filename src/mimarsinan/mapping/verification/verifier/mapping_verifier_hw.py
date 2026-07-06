from __future__ import annotations
import copy
import math
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.verification.layout_verification_packing import build_stats_from_packing_result
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_schedule_sync_count

# One layout question is asked several times per run (build-time capacity gate,
# per-model-update GUI snapshots, wizard previews) on identical shape-only
# inputs; packing is pure in those inputs, so the answer is memoized exactly.
_MEMO: "OrderedDict[Tuple, Dict[str, Any]]" = OrderedDict()
_MEMO_MAX = 8


def clear_verify_hardware_config_memo() -> None:
    _MEMO.clear()


def _memo_key(softcores, core_types, splitting, coalescing, scheduling) -> Tuple:
    return (
        tuple(
            (
                int(sc.input_count), int(sc.output_count),
                sc.threshold_group_id, sc.latency_tag, sc.segment_id, sc.name,
            )
            for sc in softcores
        ),
        tuple(
            (int(ct.get("max_axons", 0)), int(ct.get("max_neurons", 0)), int(ct.get("count", 0)))
            for ct in core_types
        ),
        bool(splitting), bool(coalescing), bool(scheduling),
    )


def verify_hardware_config(
    softcores: List[LayoutSoftCoreSpec],
    core_types: List[Dict[str, Any]],
    *,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
    allow_scheduling: bool = False,
) -> Dict[str, Any]:
    """Memoizing front of :func:`_verify_hardware_config_uncached` (LRU, deep-copied)."""
    key = _memo_key(
        softcores, core_types, allow_neuron_splitting, allow_coalescing, allow_scheduling
    )
    hit = _MEMO.get(key)
    if hit is None:
        hit = _verify_hardware_config_uncached(
            softcores,
            core_types,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            allow_scheduling=allow_scheduling,
        )
        _MEMO[key] = copy.deepcopy(hit)
        while len(_MEMO) > _MEMO_MAX:
            _MEMO.popitem(last=False)
        return hit
    _MEMO.move_to_end(key)
    return copy.deepcopy(hit)


def _verify_hardware_config_uncached(
    softcores: List[LayoutSoftCoreSpec],
    core_types: List[Dict[str, Any]],
    *,
    allow_neuron_splitting: bool = False,
    allow_coalescing: bool = False,
    allow_scheduling: bool = False,
) -> Dict[str, Any]:
    """Check whether a hardware core configuration can pack the given softcores.

    Returns a dict with ``feasible``, ``errors``, ``field_errors``,
    ``packing_result`` and ``stats`` (plus ``schedule_info`` when scheduled).
    """
    errors: List[str] = []
    field_errors: Dict[str, str] = {}

    if not softcores:
        return {
            "feasible": False,
            "errors": ["No softcores to verify against."],
            "field_errors": {},
            "packing_result": None,
        }

    if not core_types:
        return {
            "feasible": False,
            "errors": ["No core types defined. Add at least one core type."],
            "field_errors": {"core_types": "Add at least one core type."},
            "packing_result": None,
        }

    max_req_axons = max(sc.input_count for sc in softcores)
    max_req_neurons = max(sc.output_count for sc in softcores)

    hw_types: List[LayoutHardCoreType] = []
    for ct in core_types:
        hw_types.append(LayoutHardCoreType(
            max_axons=int(ct.get("max_axons", 0)),
            max_neurons=int(ct.get("max_neurons", 0)),
            count=int(ct.get("count", 0)),
        ))

    if not (allow_coalescing and allow_neuron_splitting) and not allow_scheduling:
        at_least_one_covers_largest = False
        for hw in hw_types:
            axon_ok = allow_coalescing or hw.max_axons >= max_req_axons
            neuron_ok = allow_neuron_splitting or hw.max_neurons >= max_req_neurons
            if axon_ok and neuron_ok:
                at_least_one_covers_largest = True
                break

        if not at_least_one_covers_largest:
            if allow_neuron_splitting:
                field_errors["core_types"] = (
                    f"No core type fits the largest soft core's axon count ({max_req_axons} axons). "
                    "At least one type must have max_axons >= this value (neurons will be split)."
                )
            elif allow_coalescing:
                field_errors["core_types"] = (
                    f"No core type fits the largest soft core's neuron count ({max_req_neurons} neurons). "
                    "At least one type must have max_neurons >= this value (axons will be coalesced)."
                )
            else:
                field_errors["core_types"] = (
                    f"No core type fits the largest soft core ({max_req_axons} axons, {max_req_neurons} neurons). "
                    "At least one type must have max_axons and max_neurons >= these values."
                )

    all_softcores = list(softcores)

    result = pack_layout(
        softcores=softcores,
        core_types=hw_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_coalescing=allow_coalescing,
    )

    schedule_info: Dict[str, Any] = {}
    if not result.feasible and allow_scheduling:
        from mimarsinan.mapping.support.schedule.schedule_partitioner import (
            effective_core_budget,
            estimate_passes_for_layout_validated,
        )
        budget = effective_core_budget(core_types)
        max_hw_ax = max(hw.max_axons for hw in hw_types) if hw_types else 1
        max_hw_neu = max(hw.max_neurons for hw in hw_types) if hw_types else 1

        seg_softcores: Dict[int, List[LayoutSoftCoreSpec]] = {}
        for sc in all_softcores:
            sid = sc.segment_id if sc.segment_id is not None else 0
            seg_softcores.setdefault(sid, []).append(sc)

        per_segment_passes: Dict[int, int] = {}
        per_segment_pass_lists: Dict[int, List[List[LayoutSoftCoreSpec]]] = {}
        total_pass_count = 0
        all_pass_lists: list = []
        sched_feasible = True

        for sid in sorted(seg_softcores.keys()):
            seg_scs = seg_softcores[sid]
            if budget > 0:
                n_passes, seg_pass_lists, seg_ok = estimate_passes_for_layout_validated(
                    seg_scs, budget,
                    max_hw_axons=max_hw_ax,
                    max_hw_neurons=max_hw_neu,
                    allow_coalescing=allow_coalescing,
                    allow_splitting=allow_neuron_splitting,
                    core_types=hw_types,
                )
                if not seg_ok:
                    sched_feasible = False
            else:
                n_passes, seg_pass_lists = 1, [seg_scs]
            per_segment_passes[sid] = max(n_passes, 1)
            per_segment_pass_lists[sid] = seg_pass_lists
            total_pass_count += max(n_passes, 1)
            all_pass_lists.extend(seg_pass_lists)

        best_pass_result = None
        best_pass_softcores = all_softcores
        if sched_feasible:
            for pass_scs in sorted(all_pass_lists, key=len, reverse=True):
                pr = pack_layout(
                    softcores=pass_scs,
                    core_types=hw_types,
                    allow_neuron_splitting=allow_neuron_splitting,
                    allow_coalescing=allow_coalescing,
                )
                if pr.feasible:
                    best_pass_result = pr
                    best_pass_softcores = pass_scs
                    break

        schedule_info = {
            "scheduled_feasible": sched_feasible,
            "total_passes": total_pass_count,
            "per_segment_passes": per_segment_passes,
            "per_segment_pass_lists": per_segment_pass_lists,
            "max_cores_per_pass": budget,
            "message": (
                f"Scheduled mapping: {total_pass_count} total passes across "
                f"{len(per_segment_passes)} segment(s) (cores reused between passes)."
            ) if sched_feasible else (
                "Scheduling cannot help: at least one softcore cannot be packed "
                "onto the given hardware core types (even with splitting/retries)."
            ),
        }
        if sched_feasible:
            if best_pass_result is not None and best_pass_result.feasible:
                result = best_pass_result
                softcores = best_pass_softcores
            field_errors.clear()

    if not result.feasible and not schedule_info.get("scheduled_feasible"):
        err_msg = result.error or "Hardware configuration cannot fit all soft cores."
        errors.append(err_msg)
        total_core_count = sum(int(ct.get("count", 0)) for ct in core_types)

        max_hw_ax = max(hw.max_axons for hw in hw_types) if hw_types else 1
        max_hw_neu = max(hw.max_neurons for hw in hw_types) if hw_types else 1
        per_sc_costs = []
        for sc in softcores:
            ax_f = math.ceil(sc.input_count / max_hw_ax) if allow_coalescing else 1
            neu_f = math.ceil(sc.output_count / max_hw_neu) if allow_neuron_splitting else 1
            per_sc_costs.append(ax_f * neu_f)

        if allow_scheduling:
            est_min = max(per_sc_costs) if per_sc_costs else 1
        else:
            est_min = sum(per_sc_costs)

        hint = f"Increase core counts (estimated minimum ~{est_min}) or core dimensions."
        field_errors["total_count"] = (
            f"Packing failed ({total_core_count} cores for {len(softcores)} soft cores): "
            f"{hint}"
        )

    errors = list(field_errors.values()) if field_errors else errors

    stats = build_stats_from_packing_result(
        result,
        num_original_softcores=len(all_softcores),
        softcores=all_softcores,
        core_types=hw_types,
    )
    stats_dict = stats.to_dict()

    if schedule_info.get("scheduled_feasible"):
        per_seg_passes = schedule_info.get("per_segment_passes", {})
        stats_dict["feasible"] = True
        stats_dict["schedule_pass_count"] = schedule_info.get("total_passes", 0)
        stats_dict["schedule_sync_count"] = compute_schedule_sync_count(per_seg_passes)
        stats_dict["max_cores_per_pass"] = schedule_info.get("max_cores_per_pass", 0)
        stats_dict["per_segment_passes"] = per_seg_passes

    out: Dict[str, Any] = {
        "feasible": result.feasible or schedule_info.get("scheduled_feasible", False),
        "errors": errors,
        "field_errors": field_errors,
        "packing_result": result,
        "stats": stats_dict,
    }
    if schedule_info:
        out["schedule_info"] = schedule_info
    return out
