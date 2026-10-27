from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.packing.hybrid_segment import _flush_scheduled_segment, _validate_coalescing_budget
from mimarsinan.mapping.packing.hybrid_segment_helpers import (
    _apply_reindex_to_ir_sources,
    _reindex_nodes,
)
from mimarsinan.mapping.packing.hybrid_types import HybridStage
from mimarsinan.mapping.packing.infeasibility_proofs import CoreCapacityRefusal
from mimarsinan.mapping.packing.pass_timing import PassTimingError, project_passes
from mimarsinan.mapping.packing.retimed_levels import attach_retimed_level_stages
from mimarsinan.mapping.packing.schedule_residency_plan import (
    compose_builder_program,
    depth_partition,
    mark_fit_residency,
    mark_residency_plan,
)
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.packing.type_binding import refuse_typed_multi_pass
from mimarsinan.mapping.support.period_census import ResidencySegmentPlan
from mimarsinan.mapping.layout.segmentation import (
    NeuralSegment,
    partition_ir_graph,
    segment_depth_groups,
)


class PlannedPassPackingRefusal(ToolchainRefusal, RuntimeError):
    """A pass the planner made that the packer cannot place on the declared cores, refused by name: the planner packs
    shape-only specs under their residency classes (a unique fallback per spec without the layout's), the packer the
    soft cores under theirs (the source perceptron's), so a class the planner kept apart can be merged, or not."""


def _is_capacity_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is the packer's capacity failure (``CoreCapacityRefusal``, which ``hybrid_segment`` re-raises
    with its diagnosis, W04). Any other error (``NotImplementedError``, ``RecursionError``, a defect) is not, and
    propagates unwrapped."""
    return isinstance(exc, CoreCapacityRefusal)


def _pass_levels(
    label: str, cores: list[NeuralCore], chunks: Sequence[Sequence[NeuralCore]],
) -> "list[set[int]] | None":
    """[U1 R5] The fused segment's depth levels (node ids), which every pass executes as its level units; None where the
    fused segment streams. A streaming fused segment (coalescing or psum groups keep it whole) cut with a carried wire
    refuses by name: a re-timed run's count transfer cannot carry the raster the fused segment streams."""
    groups = segment_depth_groups(cores)
    if groups is not None and len(groups) > 1:
        return [{int(n.id) for n in level} for level in groups]
    earlier: set[int] = set()
    for chunk in chunks:
        if earlier and any(int(getattr(s, "node_id", -1)) in earlier
                           for n in chunk for s in n.input_sources.flatten()):
            raise PassTimingError(
                f"pass projection: segment {label!r} streams under per-hop re-timing (coalescing or psum groups keep it "
                f"whole), and its passes carry a wire; the re-timed run's count transfer cannot carry that raster")
        earlier |= {int(n.id) for n in chunk}
    return None


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
    planned_chunks: "Sequence[Sequence[NeuralCore]] | None" = None,
    position_multiplexing: "bool | None" = None,
) -> int:
    """Flush one IR segment as the planner's passes of one shared budget.

    [U1] The composition comes from the ONE planner (residency-first,
    capacity fallback) the shape-only answer also consumes — this function
    only MATERIALIZES it: every pass carries the truthful
    ``(segment_index, pass_index)``, and an adopted residency composition is
    verified core-by-core (``mark_bank_residency``) then storage-deduped.
    [E85] ``planned_chunks``: the declared residency plan's passes of this
    segment, materialized exactly as the capacity split's. [U1 W9] The
    capacity split cuts at the specs' depth segments, as on the base; the
    planes plan the same passes given the walk's runs (``spec_runs``).
    """
    from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
    from mimarsinan.mapping.packing.canonical import declared_core_roles
    from mimarsinan.mapping.packing.hybrid_build_pool import segment_specs_for
    from mimarsinan.mapping.packing.schedule_bank_clustered import (
        dedup_resident_stage_matrices,
        mark_bank_residency,
        mark_multiplexed_residency,
        multiplexing_shares,
    )
    from mimarsinan.mapping.support.schedule.pass_planner import (
        plan_segment_passes,
    )
    from mimarsinan.mapping.support.schedule.schedule_budget import (
        effective_core_budget,
    )

    if not cores:
        return segment_index_start

    residency_applied, run_heads = False, None
    if planned_chunks is not None:
        chunks = [list(chunk) for chunk in planned_chunks]
    else:
        shares = multiplexing_shares(cores, ir_graph, segment_label_base, position_multiplexing)
        specs, spec_to_core, coalescing_group_ids = segment_specs_for(cores_config)(
            cores, ir_graph=ir_graph, hardware_bias=hardware_bias,
        )
        hw_types = [
            LayoutHardCoreType(
                max_axons=int(ct["max_axons"]),
                max_neurons=int(ct["max_neurons"]),
                count=int(ct["count"]),
                accepted_roles=roles,
            )
            for ct, roles in zip(cores_config, declared_core_roles(list(cores_config)))
        ]
        plan = plan_segment_passes(
            specs,
            effective_core_budget(list(cores_config)),
            core_types=hw_types,
            allow_coalescing=allow_coalescing,
            allow_splitting=allow_neuron_splitting,
            max_schedule_passes=max_schedule_passes,
            coalescing_group_ids=coalescing_group_ids,
            position_multiplexing=position_multiplexing,
            shares=shares,
        )
        if not plan.feasible:  # [W04] the shape-only plane's verdict and name, before anything is materialized
            if allow_coalescing:  # the specific cause first: a wide core needs more coalescing cores than declared
                _validate_coalescing_budget(cores, cores_config, allow_neuron_splitting)
            raise CoreCapacityRefusal(f"segment {segment_label_base!r} ({len(cores)} IR cores): no pass plan packs it "
                                      "on the declared cores (a soft core or a coalescing group exceeds one pass)")
        chunks = [
            [spec_to_core[id(spec)] for spec in chunk] for chunk in plan.pass_lists
        ]
        residency_applied, run_heads = plan.residency_applied, plan.run_heads
    if not chunks:
        return segment_index_start
    refuse_typed_multi_pass(segment_label_base, len(chunks), cores)
    levels = _pass_levels(segment_label_base, cores, chunks) if retimed_level_stages else None

    label_tag = (
        "res" if planned_chunks is not None
        else "pass" if residency_applied else "cap"
    )
    pass_stages: list[HybridStage] = []
    for pass_idx, chunk in enumerate(chunks):
        chunk_reindexed = (
            _reindex_nodes(chunk, all_reindex_maps)
            if all_reindex_maps else chunk
        )
        try:
            seg_stages, seg_reindex = _flush_scheduled_segment(
                current_neural=chunk_reindexed, consumed_by=consumed_by, cores_config=cores_config,
                weight_banks=weight_banks, segment_index=segment_index_start,
                segment_label=(segment_label_base if len(chunks) == 1
                               else f"{segment_label_base}_{label_tag}{pass_idx}"),
                allow_neuron_splitting=allow_neuron_splitting, allow_coalescing=allow_coalescing, pass_index=pass_idx,
            )
        except RuntimeError as exc:
            if not _is_capacity_failure(exc):
                raise
            raise PlannedPassPackingRefusal(
                f"planned pass {pass_idx} of segment {segment_label_base!r} ({len(chunk)} IR cores) does not pack on "
                f"the declared cores: {exc}") from exc
        if levels is not None:
            units = [[n for n in chunk_reindexed if int(n.id) in level] for level in levels]
            for seg_stage in seg_stages:
                attach_retimed_level_stages(
                    seg_stage,
                    current_neural=chunk_reindexed,
                    consumed_by=consumed_by,
                    weight_banks=weight_banks,
                    cores_config=cores_config,
                    allow_neuron_splitting=allow_neuron_splitting,
                    allow_coalescing=allow_coalescing,
                    level_units=[unit for unit in units if unit],
                )
        stages.extend(seg_stages)
        pass_stages.extend(seg_stages)
        all_reindex_maps.update(seg_reindex)

    if residency_applied:
        mark_bank_residency(pass_stages)
        dedup_resident_stage_matrices(pass_stages)
    if run_heads is not None:
        mark_multiplexed_residency(pass_stages, run_heads)

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
    residency_plan: "Mapping[str, Any] | None" = None,
    position_multiplexing: "bool | None" = None,
) -> "tuple[ResidencySegmentPlan, ...] | None":
    """Scheduled compilation: the planner's passes of one shared budget per segment.

    [E85] Under a declared ``residency_plan`` the whole program is planned once
    on the depth segments (one pinned budget across segments) under the refusal
    rules both planes share, each segment's passes come from that composition,
    and the executed passes are pinned and checked; the plan per segment is
    returned for the mapping. Otherwise returns None, and a program that fits is pinned whole [W03]. [U1] Every packed stage,
    single-pass segments included, is projected onto the fused timing
    (``project_passes``); under per-hop re-timing passes run as level units.
    """
    from mimarsinan.mapping.packing.hybrid_build_pool import segment_specs_for

    segment_index = 0
    segments = partition_ir_graph(ir_graph, per_hop=per_hop_neural_segments)
    builder = None
    if residency_plan is not None:
        if per_hop_neural_segments:
            raise ResidencyPlanError(
                "residency_plan plans the program's neural segments; the per-hop mapping-level split "
                "(per_hop_neural_segments, the differential reference) is not planned with it")
        segments = depth_partition(ir_graph)
        builder = compose_builder_program(
            [s for s in segments if isinstance(s, NeuralSegment)], ir_graph=ir_graph,
            cores_config=cores_config, residency_plan=residency_plan, segment_specs=segment_specs_for(cores_config),
            allow_coalescing=allow_coalescing, allow_splitting=allow_neuron_splitting,
            max_schedule_passes=max_schedule_passes,
        )
    first_stage, ordinal = len(stages), 0

    for segment in segments:
        if isinstance(segment, NeuralSegment):
            planned = None if builder is None else builder.chunks[ordinal]
            ordinal += 1
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
                planned_chunks=planned,
                position_multiplexing=position_multiplexing,
            )
        else:
            node = segment.compute_op
            op_copy = copy.copy(node)
            op_copy.input_sources = np.array(
                node.input_sources.flatten(), dtype=object,
            ).reshape(node.input_sources.shape)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))
    if not retimed_level_stages:
        project_passes(stages[first_stage:])
    neural = [s for s in stages[first_stage:] if s.kind == "neural"]
    if builder is None:
        mark_fit_residency(neural, cores_config)
        return None
    return mark_residency_plan(neural, builder, cores_config)
