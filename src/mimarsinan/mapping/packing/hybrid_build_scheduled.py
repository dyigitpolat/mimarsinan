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
    schedule_policy: str = "pool",
    max_schedule_passes: int = 8,
    retimed_level_stages: bool = False,
) -> int:
    """Flush one IR segment as passes of one shared budget.

    ``bank_clustered`` streams same-bank instances over a resident core-set
    (weights program once, verified geometry); segments outside that policy's
    class — and the default ``pool`` policy — split by capacity. Either way,
    every pass carries the truthful ``(segment_index, pass_index)``."""
    from mimarsinan.mapping.packing.hybrid_build_pool import _split_segment_by_capacity
    from mimarsinan.mapping.packing.schedule_bank_clustered import (
        dedup_resident_stage_matrices,
        mark_bank_residency,
        try_bank_clustered_passes,
    )

    if schedule_policy == "bank_clustered":
        chunks = try_bank_clustered_passes(
            cores=cores, cores_config=cores_config, weight_banks=weight_banks,
            max_schedule_passes=max_schedule_passes,
        )
        if chunks is not None:
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
                        else f"{segment_label_base}_pass{pass_idx}"
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
            mark_bank_residency(pass_stages)
            dedup_resident_stage_matrices(pass_stages)
            return segment_index_start + 1

    sub_segments = _split_segment_by_capacity(
        cores,
        cores_config,
        allow_coalescing=allow_coalescing,
        allow_neuron_splitting=allow_neuron_splitting,
        ir_graph=ir_graph,
        hardware_bias=hardware_bias,
    )
    if not sub_segments:
        return segment_index_start

    for sub_idx, sub_cores in enumerate(sub_segments):
        label = segment_label_base if len(sub_segments) == 1 else f"{segment_label_base}_cap{sub_idx}"
        sub_cores_reindexed = (
            _reindex_nodes(sub_cores, all_reindex_maps) if all_reindex_maps else sub_cores
        )
        seg_stages, seg_reindex = _flush_scheduled_segment(
            current_neural=sub_cores_reindexed,
            consumed_by=consumed_by,
            cores_config=cores_config,
            weight_banks=weight_banks,
            segment_index=segment_index_start,
            segment_label=label,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            pass_index=sub_idx,
        )
        if retimed_level_stages:
            for seg_stage in seg_stages:
                attach_retimed_level_stages(
                    seg_stage,
                    current_neural=sub_cores_reindexed,
                    consumed_by=consumed_by,
                    weight_banks=weight_banks,
                    cores_config=cores_config,
                    allow_neuron_splitting=allow_neuron_splitting,
                    allow_coalescing=allow_coalescing,
                )
        stages.extend(seg_stages)
        all_reindex_maps.update(seg_reindex)

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
    schedule_policy: str = "pool",
    max_schedule_passes: int = 8,
) -> None:
    """Scheduled compilation: passes of one shared budget per segment."""
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
                schedule_policy=schedule_policy,
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
