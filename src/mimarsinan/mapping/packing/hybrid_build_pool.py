from __future__ import annotations

import copy
from typing import Iterable, Sequence

import numpy as np

from mimarsinan.mapping.support.activation_scales import (
    compute_node_buffer_scales as _compute_node_buffer_scales,
    compute_node_input_scales as _compute_node_input_activation_scales,
    compute_node_output_scales as _compute_node_activation_scales,
)
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.ir.dynamic_bank import carry_bindings
from mimarsinan.mapping.packing.softcore import HardCore
from mimarsinan.mapping.packing.hybrid_segment import _flush_neural_segment
from mimarsinan.mapping.packing.hybrid_segment_helpers import (
    _apply_reindex_to_ir_sources,
    _make_available_hardware_cores,
    _reindex_nodes,
)
from mimarsinan.mapping.packing.hybrid_build_scheduled import _build_scheduled
from mimarsinan.mapping.packing.hybrid_build_fit import fit_program
from mimarsinan.mapping.packing.schedule_residency_plan import mark_fit_residency
from mimarsinan.mapping.packing.pass_timing import project_passes
from mimarsinan.mapping.packing.phase_channels import node_phase_channels
from mimarsinan.mapping.packing.retimed_levels import attach_retimed_level_stages
from mimarsinan.mapping.packing.segment_specs import (  # moved there: re-exported for the scheduler and tests
    _segment_specs as _segment_specs, segment_specs_for as segment_specs_for,
)
from mimarsinan.mapping.packing.hybrid_types import HybridHardCoreMapping, HybridStage
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENCY_PLAN
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.platform.platform_constraints import pool_sign_granularity
from mimarsinan.mapping.layout.segmentation import (
    NeuralSegment,
    partition_ir_graph,
)


def build_hybrid_hard_core_mapping(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    strategy: MappingStrategy | None = None,
    per_hop_neural_segments: bool = False,
    retimed_level_stages: bool = False,
    max_schedule_passes: int = 8,
) -> HybridHardCoreMapping:
    """Compile a unified IRGraph into a HybridHardCoreMapping.

    Coalesce / split / schedule passes are governed by the resolved
    :class:`MappingStrategy`; when omitted, the all-permissions-off strategy is
    used. ``retimed_level_stages`` keeps segments FUSED and attaches per-depth
    execution stages to each multi-level neural stage (C3 count-exact
    re-timing, executor-owned). ``per_hop_neural_segments`` is the historical
    mapping-level split, retained as the differential reference.
    """
    from mimarsinan.mapping.pruning.ir_segmentation import build_ir_consumed_by

    if strategy is None:
        strategy = MappingStrategy.resolve(ChipCapabilities())

    consumed_by = build_ir_consumed_by(ir_graph)

    stages: list[HybridStage] = []

    all_reindex_maps: dict[int, dict[int, int]] = {}
    residency = None

    def single_pool(stages_out: list, maps_out: dict) -> None:
        _build_single_pool(ir_graph=ir_graph, cores_config=cores_config, consumed_by=consumed_by, stages=stages_out,
                           all_reindex_maps=maps_out, allow_neuron_splitting=strategy.allow_neuron_splitting,
                           allow_coalescing=strategy.allow_coalescing, per_hop_neural_segments=per_hop_neural_segments,
                           retimed_level_stages=retimed_level_stages)

    if strategy.allow_scheduling:
        residency = _build_scheduled(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=strategy.allow_neuron_splitting,
            allow_coalescing=strategy.allow_coalescing,
            per_hop_neural_segments=per_hop_neural_segments,
            retimed_level_stages=retimed_level_stages,
            max_schedule_passes=max_schedule_passes,
            residency_plan=strategy.capabilities.residency_plan,
            position_multiplexing=strategy.capabilities.position_multiplexing,
            pipelined_multiplexing=strategy.capabilities.pipelined_multiplexing,
            layer_replication=strategy.capabilities.layer_replication,
            wavefront_group_cores=strategy.capabilities.wavefront_group_cores,
        )
        # [fix_b, fix_c] the fit rule (U1 and every declared plan but stateless): the unscheduled pack holds unwritten
        specs = _segment_specs(list(ir_graph.get_neural_cores()), ir_graph=ir_graph)[0]
        all_reindex_maps, residency = fit_program(
            single_pool, stages, all_reindex_maps, residency, residency_plan=strategy.capabilities.residency_plan,
            cells=sum(int(s.input_count) * int(s.output_count) for s in specs), cores_config=cores_config)
    else:
        single_pool(stages, all_reindex_maps)

    if not stages:
        raise ValueError("Cannot build HybridHardCoreMapping: IRGraph has no stages.")

    output_sources = ir_graph.output_sources.copy()
    _apply_reindex_to_ir_sources(output_sources, all_reindex_maps)

    mapping = HybridHardCoreMapping(
        stages=stages,
        output_sources=output_sources,
        node_activation_scales=_compute_node_activation_scales(ir_graph),
        node_input_activation_scales=_compute_node_input_activation_scales(ir_graph),
        node_buffer_scales=_compute_node_buffer_scales(ir_graph),
        node_phase_channels=node_phase_channels(ir_graph, all_reindex_maps),
    )
    if residency is not None:
        setattr(mapping, SCHEDULE_RESIDENCY_PLAN, residency)
    return carry_bindings(mapping, ir_graph)


def build_identity_hybrid_mapping(
    *, ir_graph: IRGraph, per_hop_neural_segments: bool = False,
    retimed_level_stages: bool = False, exposed: Iterable[int] = (),
) -> HybridHardCoreMapping:
    """Compile an IRGraph into a 1:1 NeuralCore→HardCore hybrid program (no pool/pad/reindex/coalesce/split).

    Carries pure IR semantics; the rung-2 SCM gate executor the packed builder
    must be value-preserving against (so it shares the per-hop segmentation
    decision with the packed build). ``exposed`` names IR nodes every stage
    also writes out (a twin certificate's backend carries, WI-5e); a node
    already read outside its stage is written out anyway, so the default
    program is unchanged.
    """
    from mimarsinan.mapping.pruning.ir_segmentation import _FINAL_OUTPUT_SENTINEL, build_ir_consumed_by

    consumed_by = build_ir_consumed_by(ir_graph)
    for node_id in exposed:
        consumed_by[int(node_id)].add(_FINAL_OUTPUT_SENTINEL)

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
        per_hop_neural_segments=per_hop_neural_segments,
        retimed_level_stages=retimed_level_stages,
    )

    if not stages:
        raise ValueError("Cannot build identity hybrid mapping: IRGraph has no stages.")

    output_sources = ir_graph.output_sources.copy()
    _apply_reindex_to_ir_sources(output_sources, all_reindex_maps)

    return carry_bindings(HybridHardCoreMapping(
        stages=stages,
        output_sources=output_sources,
        node_activation_scales=_compute_node_activation_scales(ir_graph),
        node_input_activation_scales=_compute_node_input_activation_scales(ir_graph),
        node_buffer_scales=_compute_node_buffer_scales(ir_graph),
        node_phase_channels=node_phase_channels(ir_graph, all_reindex_maps),
    ), ir_graph)


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
    per_hop_neural_segments: bool = False,
    retimed_level_stages: bool = False,
) -> None:
    """Single-shared-pool compilation path (``identity=True``: no pool). [U1] A packed stage runs at its identity
    stage's timing (``project_passes``: pins where packing moves an edge, a refusal by name where it co-packs soft
    cores the identity runs at two latencies); the identity is the timing reference and re-timed stages run levels."""
    shared_pool: list[HardCore] = (
        [] if identity else _make_available_hardware_cores(cores_config)
    )
    sign_granularity = pool_sign_granularity(cores_config)
    first_stage = len(stages)

    for segment in partition_ir_graph(ir_graph, per_hop=per_hop_neural_segments):
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
                sign_granularity=sign_granularity,
            )
            if retimed_level_stages:
                attach_retimed_level_stages(
                    stage,
                    current_neural=current_neural,
                    consumed_by=consumed_by,
                    weight_banks=ir_graph.weight_banks,
                    cores_config=cores_config,
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
    if not identity and not retimed_level_stages:
        project_passes(stages[first_stage:])
    if not identity:  # [W03] one shared pool holds every segment side by side: the planner pins the fit
        mark_fit_residency([s for s in stages[first_stage:] if s.kind == "neural"], cores_config)
