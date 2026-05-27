from __future__ import annotations

import copy
from collections import defaultdict
from typing import Sequence

import numpy as np

from mimarsinan.mapping.activation_scales import (
    compute_node_input_scales as _compute_node_input_activation_scales,
    compute_node_output_scales as _compute_node_activation_scales,
)
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.softcore_mapping import HardCore
from mimarsinan.mapping.packing.hybrid_segment import (
    _apply_reindex_to_ir_sources,
    _flush_neural_segment,
    _flush_scheduled_segment,
    _make_available_hardware_cores,
    _reindex_nodes,
)
from mimarsinan.mapping.packing.hybrid_types import HybridHardCoreMapping, HybridStage


def build_hybrid_hard_core_mapping(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    allow_neuron_splitting: bool = False,
    allow_scheduling: bool = False,
    allow_coalescing: bool = False,
) -> HybridHardCoreMapping:
    """Compile a unified IRGraph into a HybridHardCoreMapping."""

    from mimarsinan.mapping.ir_segmentation import build_ir_consumed_by

    consumed_by = build_ir_consumed_by(ir_graph)

    stages: list[HybridStage] = []

    all_reindex_maps: dict[int, dict[int, int]] = {}

    if allow_scheduling:
        _build_scheduled(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
        )
    else:
        _build_single_pool(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=allow_neuron_splitting,
        )

    if not stages:
        raise ValueError("Cannot build HybridHardCoreMapping: IRGraph has no stages.")

    output_sources = ir_graph.output_sources.copy()
    _apply_reindex_to_ir_sources(output_sources, all_reindex_maps)

    node_activation_scales = _compute_node_activation_scales(ir_graph)
    node_input_activation_scales = _compute_node_input_activation_scales(ir_graph)

    return HybridHardCoreMapping(
        stages=stages,
        output_sources=output_sources,
        node_activation_scales=node_activation_scales,
        node_input_activation_scales=node_input_activation_scales,
    )


def _build_single_pool(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    consumed_by: dict[int, set[int]],
    stages: list[HybridStage],
    all_reindex_maps: dict[int, dict[int, int]],
    allow_neuron_splitting: bool,
) -> None:
    """Original single-shared-pool compilation path."""
    shared_pool: list[HardCore] = _make_available_hardware_cores(cores_config)

    current_neural: list[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current_neural.append(node)
            continue

        if isinstance(node, ComputeOp):
            if current_neural:
                if all_reindex_maps:
                    current_neural = _reindex_nodes(current_neural, all_reindex_maps)
                stage, seg_reindex = _flush_neural_segment(
                    current_neural=current_neural,
                    consumed_by=consumed_by,
                    shared_pool=shared_pool,
                    weight_banks=ir_graph.weight_banks,
                    name=f"neural_segment_until:{node.name}",
                    allow_neuron_splitting=allow_neuron_splitting,
                )
                stages.append(stage)
                all_reindex_maps.update(seg_reindex)
                current_neural = []

            op_copy = copy.copy(node)
            op_copy.input_sources = np.array(
                node.input_sources.flatten(), dtype=object,
            ).reshape(node.input_sources.shape)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))
            continue

        raise TypeError(f"Unknown IR node type in hybrid compilation: {type(node)}")

    if current_neural:
        if all_reindex_maps:
            current_neural = _reindex_nodes(current_neural, all_reindex_maps)
        stage, seg_reindex = _flush_neural_segment(
            current_neural=current_neural,
            consumed_by=consumed_by,
            shared_pool=shared_pool,
            weight_banks=ir_graph.weight_banks,
            name="neural_segment_final",
            allow_neuron_splitting=allow_neuron_splitting,
        )
        stages.append(stage)
        all_reindex_maps.update(seg_reindex)


_SPLIT_FALLBACK_LOGGED = False


def _split_segment_by_capacity(
    cores: list[NeuralCore],
    cores_config: Sequence[dict],
    *,
    allow_coalescing: bool,
    allow_neuron_splitting: bool,
    ir_graph: IRGraph | None = None,
    hardware_bias: bool = False,
) -> list[list[NeuralCore]]:
    """Split IR NeuralCores by hardware capacity via split_softcores_by_capacity."""
    global _SPLIT_FALLBACK_LOGGED
    if not cores:
        return []

    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
    from mimarsinan.mapping.mapping_structure import compute_core_input_count
    from mimarsinan.mapping.schedule_partitioner import split_softcores_by_capacity

    hw_types = [
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct["count"]),
        )
        for ct in cores_config
    ]

    layout_specs = list(getattr(ir_graph, "layout_softcores", None) or [])
    use_layout = bool(layout_specs) and all(
        getattr(c, "layout_softcore_index", None) is not None for c in cores
    )

    specs: list[LayoutSoftCoreSpec] = []
    coalescing_group_ids: list[int | None] = []
    spec_to_core: dict[int, NeuralCore] = {}
    for idx, core in enumerate(cores):
        if use_layout:
            sc_idx = int(core.layout_softcore_index)  # type: ignore[arg-type]
            spec = layout_specs[sc_idx]
        else:
            if not _SPLIT_FALLBACK_LOGGED:
                import logging

                logging.getLogger(__name__).warning(
                    "Scheduled split: reconstructing LayoutSoftCoreSpec from NeuralCore "
                    "(missing IRGraph.layout_softcores); input_count may diverge from SCM."
                )
                _SPLIT_FALLBACK_LOGGED = True
            lat = int(core.latency) if core.latency is not None else 0
            pi = getattr(core, "perceptron_index", None)
            tg = int(pi) if pi is not None else -(idx + 1)
            n_sources = int(len(core.input_sources.flatten()))
            has_bias_axon = core.hardware_bias is None and any(
                getattr(s, "is_always_on", lambda: False)() for s in core.input_sources.flatten()
            )
            in_count = compute_core_input_count(
                n_sources - (1 if has_bias_axon else 0),
                has_bias=has_bias_axon,
                hardware_bias=hardware_bias,
            )
            spec = LayoutSoftCoreSpec(
                input_count=in_count,
                output_count=int(core.get_output_count()),
                threshold_group_id=tg,
                latency_tag=lat,
                segment_id=0,
                name=core.name,
            )
        specs.append(spec)
        coalescing_group_ids.append(getattr(core, "coalescing_group_id", None))
        spec_to_core[id(spec)] = core

    sub_specs = split_softcores_by_capacity(
        specs,
        hw_types,
        allow_coalescing=allow_coalescing,
        allow_splitting=allow_neuron_splitting,
        coalescing_group_ids=coalescing_group_ids if allow_coalescing else None,
    )
    return [[spec_to_core[id(s)] for s in sub] for sub in sub_specs]


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

    current_neural: list[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current_neural.append(node)
            continue

        if isinstance(node, ComputeOp):
            if current_neural:
                segment_index = _flush_scheduled_subsegments(
                    cores=current_neural,
                    consumed_by=consumed_by,
                    cores_config=cores_config,
                    weight_banks=ir_graph.weight_banks,
                    segment_index_start=segment_index,
                    segment_label_base=f"neural_segment_until:{node.name}",
                    allow_neuron_splitting=allow_neuron_splitting,
                    allow_coalescing=allow_coalescing,
                    all_reindex_maps=all_reindex_maps,
                    stages=stages,
                    ir_graph=ir_graph,
                )
                current_neural = []

            op_copy = copy.copy(node)
            op_copy.input_sources = np.array(
                node.input_sources.flatten(), dtype=object,
            ).reshape(node.input_sources.shape)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))
            continue

        raise TypeError(f"Unknown IR node type in hybrid compilation: {type(node)}")

    if current_neural:
        _flush_scheduled_subsegments(
            cores=current_neural,
            consumed_by=consumed_by,
            cores_config=cores_config,
            weight_banks=ir_graph.weight_banks,
            segment_index_start=segment_index,
            segment_label_base="neural_segment_final",
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            all_reindex_maps=all_reindex_maps,
            stages=stages,
            ir_graph=ir_graph,
        )
