"""
Mapping verification: confirm a model (native or torch-converted) can be represented
as soft-core mappings and retrieve shape/requirement information.

Step 1 in the hardware planning workflow:
  - Run LayoutIRMapping over the model to collect LayoutSoftCoreSpec shapes.
  - Optionally compare against actual IRMapping to ensure 1-1 correspondence.
  - Return a structured result usable by the greedy HW config suggester.

``verify_hardware_config`` notes
---------------------------------
- Does **not** pre-check ``total_count >= len(softcores)``.  Multiple softcores
  can be packed into a single hardware core (bin-packing), so fewer hardware
  cores than softcores is perfectly valid.  ``pack_layout`` is the sole arbiter
  of feasibility.
- The ``"total_count"`` key in ``field_errors`` only appears when packing
  genuinely fails — it is never raised as a speculative pre-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec, LayoutHardCoreType
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout_verification_stats import (
    build_stats_from_packing_result,
    compute_schedule_sync_count,
)


@dataclass
class MappingVerificationResult:
    """Result of a soft-core mapping verification pass."""

    feasible: bool
    softcores: List[LayoutSoftCoreSpec]

    # Summary statistics
    num_neural_cores: int
    max_input_size: int
    max_output_size: int
    total_area: int

    # Layout-derived count of host-side compute segments between / around
    # neural segments. Used by the wizard as the "Sync Barriers" metric.
    host_side_segment_count: int = 0

    # Compact layout-only flow summary for the wizard miniview.
    layout_preview: Optional[Dict[str, Any]] = None

    # Optional error message when feasible=False
    error: Optional[str] = None


def verify_soft_core_mapping(
    model_repr,
    max_axons: int,
    max_neurons: int,
    *,
    threshold_groups: int = 1,
    pruning_fraction: float = 0.0,
    threshold_seed: int = 0,
    allow_core_coalescing: bool = False,
    hardware_bias: bool = False,
) -> MappingVerificationResult:
    """Verify that a mapper-graph model representation can be laid out as soft cores.

    Parameters
    ----------
    model_repr:
        A ``ModelRepresentation`` (or any object exposing ``map_to_ir(mapping)``)
        for a native mimarsinan model, OR the ``mapper_repr`` obtained from a
        torch-converted ``Supermodel`` via ``supermodel.get_mapper_repr()``.
    max_axons:
        Maximum axon count per hardware core.
    max_neurons:
        Maximum neuron count per hardware core.
    threshold_groups:
        Number of threshold groups to simulate (random assignment).
    pruning_fraction:
        Expected pruning fraction (0–1). The estimator applies 80% of this
        as a random row/column reduction on each softcore.
    threshold_seed:
        RNG seed for deterministic threshold-group assignment.
    allow_core_coalescing:
        If True, enable layout-level axon tiling via coalescing (wide FC
        layers keep their full width and hardware fuses physical cores).
    hardware_bias:
        If True, bias is in a dedicated register, not an always-on axon row.

    Returns
    -------
    MappingVerificationResult
    """
    try:
        layout = LayoutIRMapping(
            max_axons=max_axons,
            max_neurons=max_neurons,
            threshold_groups=threshold_groups,
            threshold_seed=threshold_seed,
            pruning_fraction=pruning_fraction,
            allow_core_coalescing=allow_core_coalescing,
            hardware_bias=hardware_bias,
        )
        softcores = layout.collect_layout_softcores(model_repr)
    except Exception as exc:
        return MappingVerificationResult(
            feasible=False,
            softcores=[],
            num_neural_cores=0,
            max_input_size=0,
            max_output_size=0,
            total_area=0,
            host_side_segment_count=0,
            layout_preview=None,
            error=str(exc),
        )

    if not softcores:
        return MappingVerificationResult(
            feasible=False,
            softcores=[],
            num_neural_cores=0,
            max_input_size=0,
            max_output_size=0,
            total_area=0,
            host_side_segment_count=getattr(layout, "host_side_segment_count", 0),
            layout_preview=getattr(layout, "layout_preview", None),
            error="No neural cores produced by mapping — model may have no perceptron layers.",
        )

    max_in = max(sc.input_count for sc in softcores)
    max_out = max(sc.output_count for sc in softcores)
    total_area = sum(sc.area for sc in softcores)

    return MappingVerificationResult(
        feasible=True,
        softcores=softcores,
        num_neural_cores=len(softcores),
        max_input_size=max_in,
        max_output_size=max_out,
        total_area=total_area,
        host_side_segment_count=getattr(layout, "host_side_segment_count", 0),
        layout_preview=getattr(layout, "layout_preview", None),
    )


def verify_hardware_config(
    softcores: List[LayoutSoftCoreSpec],
    core_types: List[Dict[str, Any]],
    *,
    allow_neuron_splitting: bool = False,
    allow_axon_coalescing: bool = False,
    allow_scheduling: bool = False,
) -> Dict[str, Any]:
    """Check whether a hardware core configuration is sufficient for the given softcores.

    Parameters
    ----------
    softcores:
        List of ``LayoutSoftCoreSpec`` from ``verify_soft_core_mapping``.
    core_types:
        List of dicts with keys ``max_axons``, ``max_neurons``, ``count``.
    allow_neuron_splitting:
        If True, soft cores may be split along the neuron dimension during
        packing, so the dimension pre-check only requires axon coverage.
    allow_axon_coalescing:
        If True, soft cores whose input count exceeds a single core's max_axons
        are coalesced across multiple hardware cores, so the pre-check only
        requires neuron coverage.

    Returns
    -------
    dict with keys:
        - ``feasible`` (bool): True if all softcores can be packed.
        - ``errors`` (list[str]): Human-readable error messages (empty on success).
        - ``field_errors`` (dict): Per-field error hints for UI display.
          Keys: ``"max_axons"``, ``"max_neurons"``, ``"count"`` per core-type index.
        - ``packing_result``: ``LayoutPackingResult`` (or None on fatal error).
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

    # Build LayoutHardCoreType list and check feasibility of dimensions.
    # With neuron splitting, at least one core type must cover the largest
    # axon count (neurons will be split as needed).  Without splitting,
    # at least one type must cover both dimensions.
    hw_types: List[LayoutHardCoreType] = []
    for ct in core_types:
        hw_types.append(LayoutHardCoreType(
            max_axons=int(ct.get("max_axons", 0)),
            max_neurons=int(ct.get("max_neurons", 0)),
            count=int(ct.get("count", 0)),
        ))

    # Pre-check: confirm at least one core type can accept the largest softcore.
    # When both features are active any softcore is mappable regardless of dimensions
    # (splitting distributes outputs, coalescing distributes inputs), so no check needed.
    # When scheduling is enabled, coalescing/splitting fragments can be distributed
    # across passes, so dimension constraints are always satisfiable.
    # When only one feature is active (no scheduling), the non-split dimension is still a hard constraint.
    if not (allow_axon_coalescing and allow_neuron_splitting) and not allow_scheduling:
        at_least_one_covers_largest = False
        for hw in hw_types:
            axon_ok = allow_axon_coalescing or hw.max_axons >= max_req_axons
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
            elif allow_axon_coalescing:
                field_errors["core_types"] = (
                    f"No core type fits the largest soft core's neuron count ({max_req_neurons} neurons). "
                    "At least one type must have max_neurons >= this value (axons will be coalesced)."
                )
            else:
                field_errors["core_types"] = (
                    f"No core type fits the largest soft core ({max_req_axons} axons, {max_req_neurons} neurons). "
                    "At least one type must have max_axons and max_neurons >= these values."
                )

    # Attempt greedy packing — this is the precise feasibility check.
    # (We do NOT pre-check total_count < len(softcores): multiple softcores can
    # share a single hardware core, so far fewer than len(softcores) cores may suffice.)
    all_softcores = list(softcores)  # preserve full list for stats

    result = pack_layout(
        softcores=softcores,
        core_types=hw_types,
        allow_neuron_splitting=allow_neuron_splitting,
        allow_axon_coalescing=allow_axon_coalescing,
    )

    # When scheduling is enabled and single-pass packing fails, partition
    # softcores into schedule passes.  The partitioner validates each pass
    # with real typed packing (including fragment expansion for softcores
    # whose coalescing/splitting fragments exceed available cores).  Every
    # validated pass is guaranteed to pack on the real hardware.
    schedule_info: Dict[str, Any] = {}
    if not result.feasible and allow_scheduling:
        from mimarsinan.mapping.schedule_partitioner import (
            effective_core_budget,
            estimate_passes_for_layout_validated,
        )
        budget = effective_core_budget(core_types)
        max_hw_ax = max(hw.max_axons for hw in hw_types) if hw_types else 1
        max_hw_neu = max(hw.max_neurons for hw in hw_types) if hw_types else 1

        common_est_kwargs = dict(
            max_hw_axons=max_hw_ax,
            max_hw_neurons=max_hw_neu,
            allow_coalescing=allow_axon_coalescing,
            allow_splitting=allow_neuron_splitting,
            core_types=hw_types,
        )

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
                    seg_scs, budget, **common_est_kwargs,
                )
                if not seg_ok:
                    sched_feasible = False
            else:
                n_passes, seg_pass_lists = 1, [seg_scs]
            per_segment_passes[sid] = max(n_passes, 1)
            per_segment_pass_lists[sid] = seg_pass_lists
            total_pass_count += max(n_passes, 1)
            all_pass_lists.extend(seg_pass_lists)

        # Pack the busiest validated pass for representative utilization stats.
        best_pass_result = None
        best_pass_softcores = all_softcores
        if sched_feasible:
            for pass_scs in sorted(all_pass_lists, key=len, reverse=True):
                try:
                    pr = pack_layout(
                        softcores=pass_scs,
                        core_types=hw_types,
                        allow_neuron_splitting=allow_neuron_splitting,
                        allow_axon_coalescing=allow_axon_coalescing,
                    )
                    if pr.feasible:
                        best_pass_result = pr
                        best_pass_softcores = pass_scs
                        break
                except Exception:
                    continue

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

        import math
        max_hw_ax = max(hw.max_axons for hw in hw_types) if hw_types else 1
        max_hw_neu = max(hw.max_neurons for hw in hw_types) if hw_types else 1
        per_sc_costs = []
        for sc in softcores:
            ax_f = math.ceil(sc.input_count / max_hw_ax) if allow_axon_coalescing else 1
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
