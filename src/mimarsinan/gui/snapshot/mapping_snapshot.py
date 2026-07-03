"""GUI snapshot module."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.snapshot.util.helpers import _histogram
from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.heatmap import _make_heatmap_producer
from mimarsinan.gui.snapshot.ir_graph.ir_graph_resources import (
    _group_consecutive_compute_stages,
    _make_per_core_connectivity_producer,
    _make_segment_spans_extractor,
)

RESOURCE_KIND_IR_CORE_HEATMAP = "ir_core_heatmap"
RESOURCE_KIND_IR_CORE_PRE_PRUNING = "ir_core_pre_pruning"
RESOURCE_KIND_IR_CORE_BIAS = "ir_core_bias"
RESOURCE_KIND_IR_BANK_HEATMAP = "ir_bank_heatmap"
RESOURCE_KIND_HARD_CORE_HEATMAP = "hard_core_heatmap"
RESOURCE_KIND_CONNECTIVITY = "connectivity"
RESOURCE_KIND_PRUNING_LAYER_HEATMAP = "pruning_layer_heatmap"

LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"

def snapshot_mapping_performance_planned(
    model: Any,
    platform_constraints: dict | None,
    *,
    input_shape: tuple | list | None = None,
    num_classes: int | None = None,
) -> dict | None:
    """Return wizard-shaped Mapping Performance stats for the built model."""
    if model is None or not platform_constraints:
        return None
    cores = platform_constraints.get("cores") or []
    if not cores:
        return None
    try:
        from mimarsinan.mapping.verification.wizard_layout_verify import (
            model_repr_from_model,
            verify_planned_mapping_performance,
        )
    except Exception:
        logger.debug("wizard_layout_verify not importable", exc_info=True)
        return None

    model_repr = model_repr_from_model(
        model, input_shape=input_shape, num_classes=num_classes or 10
    )
    if model_repr is None:
        return None
    try:
        return verify_planned_mapping_performance(model_repr, platform_constraints)
    except Exception:
        logger.debug("verify_planned_mapping_performance failed", exc_info=True)
        return None


def snapshot_mapping_performance_real(mapping: Any) -> dict | None:
    """Derive wizard-shaped Mapping Performance stats from a real mapping."""
    from mimarsinan.mapping.verification.layout_verification_hybrid import stats_dict_from_hybrid_mapping

    return stats_dict_from_hybrid_mapping(mapping)


def snapshot_hard_core_mapping(mapping: Any) -> tuple[dict, list[ResourceDescriptor]]:
    """Extract utilization, packing, stage flow, and per-core detail."""
    stages_info: list[dict] = []
    all_core_utils: list[dict] = []
    neural_segment_idx = 0
    descriptors: list[ResourceDescriptor] = []

    for i, stage in enumerate(mapping.stages):
        stage_info: dict = {"index": i, "kind": stage.kind, "name": stage.name}

        if stage.kind == "neural" and stage.hard_core_mapping is not None:
            hcm = stage.hard_core_mapping
            seg_idx = neural_segment_idx
            stage_info["segment_index"] = seg_idx
            neural_segment_idx += 1

            if stage.schedule_pass_index is not None:
                stage_info["schedule_pass_index"] = stage.schedule_pass_index
            if stage.schedule_segment_index is not None:
                stage_info["schedule_segment_index"] = stage.schedule_segment_index

            cores_detail: list[dict] = []
            seg_spans_extractor = _make_segment_spans_extractor(hcm, seg_idx)
            for ci, core in enumerate(hcm.cores):
                used_axons = core.axons_per_core - core.available_axons
                used_neurons = core.neurons_per_core - core.available_neurons
                total = core.axons_per_core * core.neurons_per_core
                used = used_axons * used_neurons
                utilization = used / total if total > 0 else 0.0
                core_d: dict = {
                    "core_index": ci,
                    "axons_per_core": core.axons_per_core,
                    "neurons_per_core": core.neurons_per_core,
                    "used_axons": used_axons,
                    "used_neurons": used_neurons,
                    "display_axons": core.axons_per_core,
                    "display_neurons": core.neurons_per_core,
                    "utilization": utilization,
                    "threshold": float(core.threshold) if core.threshold is not None else None,
                    "latency": core.latency,
                }
                try:
                    mat = core.core_matrix
                    rid = f"seg/{seg_idx}/core/{ci}"
                    core_d["has_heatmap"] = True
                    core_d["heatmap_resource"] = {
                        "kind": RESOURCE_KIND_HARD_CORE_HEATMAP,
                        "rid": rid,
                    }
                    core_d["heatmap_axons"] = int(mat.shape[0])
                    core_d["heatmap_neurons"] = int(mat.shape[1])
                    descriptors.append(ResourceDescriptor(
                        kind=RESOURCE_KIND_HARD_CORE_HEATMAP,
                        rid=rid,
                        producer=_make_heatmap_producer(mat, copy=False),
                        media_type="image/png",
                    ))
                except Exception:
                    logger.debug("Failed to register heatmap for hard core %d", ci, exc_info=True)
                conn_rid = f"seg/{seg_idx}/core/{ci}"
                core_d["has_connectivity"] = True
                core_d["connectivity_resource"] = {
                    "kind": RESOURCE_KIND_CONNECTIVITY,
                    "rid": conn_rid,
                }
                descriptors.append(ResourceDescriptor(
                    kind=RESOURCE_KIND_CONNECTIVITY,
                    rid=conn_rid,
                    producer=_make_per_core_connectivity_producer(seg_spans_extractor, ci),
                    media_type="application/json",
                ))
                placements = getattr(hcm, "soft_core_placements_per_hard_core", None)
                if placements is None:
                    raise ValueError(
                        "Hard core mapping is missing soft-core traceability data "
                        "(soft_core_placements_per_hard_core). Re-run the Hard Core Mapping step to regenerate the mapping."
                    )
                if ci >= len(placements):
                    raise ValueError(
                        f"Hard core mapping traceability inconsistent: core index {ci} has no placement list "
                        "(soft_core_placements_per_hard_core). Re-run the Hard Core Mapping step."
                    )
                pl_list = placements[ci]
                total_area = core.axons_per_core * core.neurons_per_core
                mapped_placements = []
                for pl in pl_list:
                    pl_copy = dict(pl)
                    ax, nu = pl_copy.get("axons", 0), pl_copy.get("neurons", 0)
                    pl_copy["utilization_frac"] = (ax * nu / total_area) if total_area > 0 else 0.0
                    mapped_placements.append(pl_copy)
                core_d["mapped_placements"] = mapped_placements
                core_d["constituent_count"] = len(pl_list)
                fused_axons = getattr(core, "fused_component_axons", None)
                if fused_axons:
                    boundaries = [0]
                    for c in fused_axons:
                        boundaries.append(boundaries[-1] + c)
                    core_d["fused_axon_boundaries"] = boundaries
                    core_d["fused_component_count"] = len(fused_axons)
                cores_detail.append(core_d)
                all_core_utils.append(core_d)
            stage_info["num_cores"] = len(hcm.cores)
            stage_info["cores"] = cores_detail
            stage_info["has_connectivity"] = True

            try:
                stage_info["input_map"] = [
                    {"node_id": s.node_id, "offset": s.offset, "size": s.size}
                    for s in stage.input_map
                ]
                stage_info["output_map"] = [
                    {"node_id": s.node_id, "offset": s.offset, "size": s.size}
                    for s in stage.output_map
                ]
            except Exception:
                logger.debug("Failed to extract io_map for stage %d", i, exc_info=True)

        elif stage.kind == "compute" and stage.compute_op is not None:
            stage_info["op_type"] = stage.compute_op.op_type
            stage_info["op_name"] = stage.compute_op.name
            stage_info["input_shape"] = list(stage.compute_op.input_shape) if getattr(stage.compute_op, "input_shape", None) is not None else None
            stage_info["output_shape"] = list(stage.compute_op.output_shape) if getattr(stage.compute_op, "output_shape", None) is not None else None
            stage_info["is_barrier"] = True

        stages_info.append(stage_info)

    stages_info = _group_consecutive_compute_stages(stages_info)

    core_reuse = _compute_core_reuse(stages_info)
    global_core_layout = _compute_global_core_layout(stages_info)

    utilizations = [c["utilization"] for c in all_core_utils]
    summary = {
        "num_stages": len(mapping.stages),
        "num_neural_segments": len(mapping.get_neural_segments()),
        "num_compute_ops": len(mapping.get_compute_ops()),
        "total_cores": len(all_core_utils),
        "stages": stages_info,
        "core_reuse": core_reuse,
        "global_core_layout": global_core_layout,
        "utilization_histogram": _histogram(np.array(utilizations)) if utilizations else None,
        "mean_utilization": float(np.mean(utilizations)) if utilizations else 0.0,
    }
    return summary, descriptors


def _compute_global_core_layout(stages: list[dict]) -> list[dict]:
    """Compute the global hardware core layout across all neural segments."""
    dim_max_count: dict[tuple[int, int], int] = defaultdict(int)
    for s in stages:
        if s["kind"] != "neural" or "cores" not in s:
            continue
        seg_counts: dict[tuple[int, int], int] = defaultdict(int)
        for c in s["cores"]:
            dim = (c["axons_per_core"], c["neurons_per_core"])
            seg_counts[dim] += 1
        for dim, cnt in seg_counts.items():
            dim_max_count[dim] = max(dim_max_count[dim], cnt)
    return [
        {"axons_per_core": a, "neurons_per_core": n, "count": cnt}
        for (a, n), cnt in sorted(dim_max_count.items())
    ]


def _compute_core_reuse(stages: list[dict]) -> dict:
    """Compute per-core-dimension reuse across neural segments."""
    dim_to_segments: dict[str, list[int]] = defaultdict(list)
    for s in stages:
        if s["kind"] != "neural" or "cores" not in s:
            continue
        seg_idx = s.get("segment_index", s["index"])
        for c in s["cores"]:
            dim_key = f"{c['axons_per_core']}x{c['neurons_per_core']}"
            if seg_idx not in dim_to_segments[dim_key]:
                dim_to_segments[dim_key].append(seg_idx)
    return {
        "core_configs": [
            {"dimensions": k, "segments": v, "num_segments": len(v)}
            for k, v in dim_to_segments.items()
        ]
    }


