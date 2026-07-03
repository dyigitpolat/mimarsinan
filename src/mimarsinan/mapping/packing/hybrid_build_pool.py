from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from mimarsinan.mapping.support.activation_scales import (
    compute_node_input_scales as _compute_node_input_activation_scales,
    compute_node_output_scales as _compute_node_activation_scales,
)
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.packing.softcore import HardCore
from mimarsinan.mapping.packing.hybrid_segment import _flush_neural_segment
from mimarsinan.mapping.packing.hybrid_segment_helpers import (
    _apply_reindex_to_ir_sources,
    _make_available_hardware_cores,
    _reindex_nodes,
)
from mimarsinan.mapping.packing.hybrid_build_scheduled import _build_scheduled
from mimarsinan.mapping.packing.hybrid_types import HybridHardCoreMapping, HybridStage
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.layout.segmentation import (
    NeuralSegment,
    partition_ir_graph,
)


def build_hybrid_hard_core_mapping(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    strategy: MappingStrategy | None = None,
) -> HybridHardCoreMapping:
    """Compile a unified IRGraph into a HybridHardCoreMapping.

    Coalesce / split / schedule passes are governed by the resolved
    :class:`MappingStrategy`; when omitted, the all-permissions-off strategy is used.
    """
    from mimarsinan.mapping.pruning.ir_segmentation import build_ir_consumed_by

    if strategy is None:
        strategy = MappingStrategy.resolve(ChipCapabilities())

    consumed_by = build_ir_consumed_by(ir_graph)

    stages: list[HybridStage] = []

    all_reindex_maps: dict[int, dict[int, int]] = {}

    if strategy.allow_scheduling:
        _build_scheduled(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=strategy.allow_neuron_splitting,
            allow_coalescing=strategy.allow_coalescing,
        )
    else:
        _build_single_pool(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=strategy.allow_neuron_splitting,
            allow_coalescing=strategy.allow_coalescing,
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


def build_identity_hybrid_mapping(*, ir_graph: IRGraph) -> HybridHardCoreMapping:
    """Compile an IRGraph into a 1:1 NeuralCore→HardCore hybrid program (no pool/pad/reindex/coalesce/split).

    Carries pure IR semantics; the rung-2 SCM gate executor the packed builder
    must be value-preserving against.
    """
    from mimarsinan.mapping.pruning.ir_segmentation import build_ir_consumed_by

    consumed_by = build_ir_consumed_by(ir_graph)

    stages: list[HybridStage] = []
    all_reindex_maps: dict[int, dict[int, int]] = {}
    _build_single_pool(
        ir_graph=ir_graph,
        cores_config=(),
        consumed_by=consumed_by,
        stages=stages,
        all_reindex_maps=all_reindex_maps,
        allow_neuron_splitting=False,
        identity=True,
    )

    if not stages:
        raise ValueError("Cannot build identity hybrid mapping: IRGraph has no stages.")

    output_sources = ir_graph.output_sources.copy()
    _apply_reindex_to_ir_sources(output_sources, all_reindex_maps)

    return HybridHardCoreMapping(
        stages=stages,
        output_sources=output_sources,
        node_activation_scales=_compute_node_activation_scales(ir_graph),
        node_input_activation_scales=_compute_node_input_activation_scales(ir_graph),
    )


def _build_single_pool(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    consumed_by: dict[int, set[int]],
    stages: list[HybridStage],
    all_reindex_maps: dict[int, dict[int, int]],
    allow_neuron_splitting: bool,
    allow_coalescing: bool = True,
    identity: bool = False,
) -> None:
    """Single-shared-pool compilation path (``identity=True``: no pool)."""
    shared_pool: list[HardCore] = (
        [] if identity else _make_available_hardware_cores(cores_config)
    )

    for segment in partition_ir_graph(ir_graph):
        if isinstance(segment, NeuralSegment):
            current_neural = segment.nodes
            if all_reindex_maps:
                current_neural = _reindex_nodes(current_neural, all_reindex_maps)
            stage, seg_reindex = _flush_neural_segment(
                current_neural=current_neural,
                consumed_by=consumed_by,
                shared_pool=shared_pool,
                weight_banks=ir_graph.weight_banks,
                name=segment.label,
                allow_neuron_splitting=allow_neuron_splitting,
                allow_coalescing=allow_coalescing,
                identity=identity,
            )
            stages.append(stage)
            all_reindex_maps.update(seg_reindex)
        else:
            node = segment.compute_op
            op_copy = copy.copy(node)
            op_copy.input_sources = np.array(
                node.input_sources.flatten(), dtype=object,
            ).reshape(node.input_sources.shape)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))


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
    from mimarsinan.mapping.layout.softcore_spec_adapter import spec_from_neural_core
    from mimarsinan.mapping.support.schedule.schedule_partitioner import split_softcores_by_capacity

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
            spec = spec_from_neural_core(
                core,
                hardware_bias=hardware_bias,
                fallback_threshold_group_id=-(idx + 1),
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

