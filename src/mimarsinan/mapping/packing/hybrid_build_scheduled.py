from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.packing.hybrid_segment import _flush_scheduled_segment
from mimarsinan.mapping.packing.hybrid_segment_helpers import (
    _apply_reindex_to_ir_sources,
    _reindex_nodes,
)
from mimarsinan.mapping.packing.hybrid_types import HybridStage
from mimarsinan.mapping.packing.retimed_levels import attach_retimed_level_stages
from mimarsinan.mapping.layout.segmentation import (
    NeuralSegment,
    partition_ir_graph,
)


def _flush_scheduled_subsegments(
    *,
    cores: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    cores_config: Sequence[dict],
    weight_banks: dict,
    segment_index_start: int,
    segment_label_base: str,
    allow_neuron_splitting: bool,
    allow_coalescing: bool,
    all_reindex_maps: dict[int, dict[int, int]],
    stages: list[HybridStage],
    ir_graph: IRGraph | None = None,
    hardware_bias: bool = False,
    max_schedule_passes: int = 8,
    retimed_level_stages: bool = False,
) -> int:
    """Flush one IR segment as the planner's passes of one shared budget.

    [U1] The composition comes from the ONE planner (residency-first,
    capacity fallback) the shape-only answer also consumes — this function
    only MATERIALIZES it: every pass carries the truthful
    ``(segment_index, pass_index)``, and an adopted residency composition is
    verified core-by-core (``mark_bank_residency``) then storage-deduped.
    """
    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
    from mimarsinan.mapping.packing.hybrid_build_pool import _segment_specs
    from mimarsinan.mapping.packing.schedule_bank_clustered import (
        dedup_resident_stage_matrices,
        mark_bank_residency,
    )
    from mimarsinan.mapping.support.schedule.pass_planner import (
        plan_segment_passes,
    )
    from mimarsinan.mapping.support.schedule.schedule_budget import (
        effective_core_budget,
    )

    if not cores:
        return segment_index_start

    specs, spec_to_core, coalescing_group_ids = _segment_specs(
        cores, ir_graph=ir_graph, hardware_bias=hardware_bias,
    )
    hw_types = [
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct["count"]),
        )
        for ct in cores_config
    ]
    plan = plan_segment_passes(
        specs,
        effective_core_budget(list(cores_config)),
        core_types=hw_types,
        allow_coalescing=allow_coalescing,
        allow_splitting=allow_neuron_splitting,
        max_schedule_passes=max_schedule_passes,
        coalescing_group_ids=coalescing_group_ids,
    )
    chunks = [
        [spec_to_core[id(spec)] for spec in chunk] for chunk in plan.pass_lists
    ]
    if not chunks:
        return segment_index_start

    label_tag = "pass" if plan.residency_applied else "cap"
    pass_stages: list[HybridStage] = []
    for pass_idx, chunk in enumerate(chunks):
        chunk_reindexed = (
            _reindex_nodes(chunk, all_reindex_maps)
            if all_reindex_maps else chunk
        )
        seg_stages, seg_reindex = _flush_scheduled_segment(
            current_neural=chunk_reindexed,
            consumed_by=consumed_by,
            cores_config=cores_config,
            weight_banks=weight_banks,
            segment_index=segment_index_start,
            segment_label=(
                segment_label_base if len(chunks) == 1
                else f"{segment_label_base}_{label_tag}{pass_idx}"
            ),
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            pass_index=pass_idx,
        )
        if retimed_level_stages:
            for seg_stage in seg_stages:
                attach_retimed_level_stages(
                    seg_stage,
                    current_neural=chunk_reindexed,
                    consumed_by=consumed_by,
                    weight_banks=weight_banks,
                    cores_config=cores_config,
                    allow_neuron_splitting=allow_neuron_splitting,
                    allow_coalescing=allow_coalescing,
                )
        stages.extend(seg_stages)
        pass_stages.extend(seg_stages)
        all_reindex_maps.update(seg_reindex)

    if plan.residency_applied:
        mark_bank_residency(pass_stages)
        dedup_resident_stage_matrices(pass_stages)

    return segment_index_start + 1


def _build_scheduled(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    consumed_by: dict[int, set[int]],
    stages: list[HybridStage],
    all_reindex_maps: dict[int, dict[int, int]],
    allow_neuron_splitting: bool,
    allow_coalescing: bool = False,
    per_hop_neural_segments: bool = False,
    retimed_level_stages: bool = False,
    max_schedule_passes: int = 8,
) -> None:
    """Scheduled compilation: the planner's passes of one shared budget per segment."""
    segment_index = 0

    for segment in partition_ir_graph(ir_graph, per_hop=per_hop_neural_segments):
        if isinstance(segment, NeuralSegment):
            segment_index = _flush_scheduled_subsegments(
                cores=segment.nodes,
                consumed_by=consumed_by,
                cores_config=cores_config,
                weight_banks=ir_graph.weight_banks,
                segment_index_start=segment_index,
                segment_label_base=segment.label,
                allow_neuron_splitting=allow_neuron_splitting,
                allow_coalescing=allow_coalescing,
                all_reindex_maps=all_reindex_maps,
                stages=stages,
                ir_graph=ir_graph,
                max_schedule_passes=max_schedule_passes,
                retimed_level_stages=retimed_level_stages,
            )
        else:
            node = segment.compute_op
            op_copy = copy.copy(node)
            op_copy.input_sources = np.array(
                node.input_sources.flatten(), dtype=object,
            ).reshape(node.input_sources.shape)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))
