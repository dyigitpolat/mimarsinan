from __future__ import annotations

import copy
from collections import defaultdict
from typing import Sequence

import numpy as np

from mimarsinan.mapping.support.activation_scales import (
    compute_node_input_scales as _compute_node_input_activation_scales,
    compute_node_output_scales as _compute_node_activation_scales,
)
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.packing.softcore import HardCore
from mimarsinan.mapping.packing.hybrid_segment import (
    _apply_reindex_to_ir_sources,
    _flush_neural_segment,
    _flush_scheduled_segment,
    _make_available_hardware_cores,
    _reindex_nodes,
)
from mimarsinan.mapping.packing.hybrid_types import HybridHardCoreMapping, HybridStage
from mimarsinan.mapping.layout.segmentation import (
    HostSegment,
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
) -> int:
    """Flush one IR segment; split by capacity when scheduling."""
    from mimarsinan.mapping.packing.hybrid_build_pool import _split_segment_by_capacity

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
            segment_index=segment_index_start + sub_idx,
            segment_label=label,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )
        stages.extend(seg_stages)
        all_reindex_maps.update(seg_reindex)

    return segment_index_start + len(sub_segments)


def _build_scheduled(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    consumed_by: dict[int, set[int]],
    stages: list[HybridStage],
    all_reindex_maps: dict[int, dict[int, int]],
    allow_neuron_splitting: bool,
    allow_coalescing: bool = False,
) -> None:
    """Scheduled compilation: fresh core pool per pass."""
    segment_index = 0

    for segment in partition_ir_graph(ir_graph):
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
            )
        else:
            node = segment.compute_op
            op_copy = copy.copy(node)
            op_copy.input_sources = np.array(
                node.input_sources.flatten(), dtype=object,
            ).reshape(node.input_sources.shape)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))
