from __future__ import annotations

import copy
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore, ir_graph_to_soft_core_mapping
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping, compact_soft_core_mapping


_FINAL_OUTPUT_SENTINEL = -999


@dataclass
class SegmentIOSlice:
    """Maps a contiguous slice of a neural segment's I/O buffer to a state-buffer entry."""
    node_id: int
    offset: int
    size: int


@dataclass
class HybridStage:
    """
    A single stage in a hybrid runtime program.

    - "neural": A HardCoreMapping that can be executed on the chip runtime.
    - "compute": A ComputeOp that must be executed as a sync barrier.

    Neural stages carry ``input_map`` / ``output_map`` metadata so the
    runtime can assemble the segment's input from a global state buffer
    and store the segment's output back into that buffer.
    """

    kind: Literal["neural", "compute"]
    name: str
    hard_core_mapping: HardCoreMapping | None = None
    compute_op: ComputeOp | None = None
    input_map: list[SegmentIOSlice] = field(default_factory=list)
    output_map: list[SegmentIOSlice] = field(default_factory=list)
    # Scheduling metadata (None when scheduling is off / single-pass segments)
    schedule_segment_index: int | None = None
    schedule_pass_index: int | None = None


@dataclass
class HybridHardCoreMapping:
    """
    A deployable *hybrid* program representation:

    neural segment (HardCoreMapping) -> ComputeOp barrier -> neural segment -> ...

    Each neural segment can be packed/codegen'ed as a standalone chip program.
    ComputeOps represent host-side (or auxiliary) computation between chip runs.

    ``output_sources`` mirrors the original IRGraph output_sources so the
    runtime can assemble the final network output from the state buffer.
    """

    stages: List[HybridStage]
    output_sources: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    def get_compute_ops(self) -> List[ComputeOp]:
        return [s.compute_op for s in self.stages if s.kind == "compute" and s.compute_op is not None]

    def get_neural_segments(self) -> List[HardCoreMapping]:
        return [s.hard_core_mapping for s in self.stages if s.kind == "neural" and s.hard_core_mapping is not None]


def _make_available_hardware_cores(cores_config: Sequence[dict]) -> list[HardCore]:
    available_hardware_cores: list[HardCore] = []
    for core_type in cores_config:
        count = int(core_type["count"])
        max_axons = int(core_type["max_axons"])
        max_neurons = int(core_type["max_neurons"])
        has_bias = bool(core_type.get("has_bias", True))
        for _ in range(count):
            available_hardware_cores.append(
                HardCore(max_axons, max_neurons, has_bias_capability=has_bias)
            )
    return available_hardware_cores


def _remap_external_sources_to_segment_inputs(
    *,
    nodes: list[NeuralCore],
    output_sources: np.ndarray,
    weight_banks: dict | None = None,
) -> tuple[IRGraph, list[SegmentIOSlice]]:
    """
    Build a neural-only IRGraph for a segment with support for
    multiple external source nodes (skip connections).

    External IRSource references (node_id >= 0, not in this segment) are
    remapped to segment-local inputs (IRSource(node_id=-2, index=...)).
    Multiple external nodes are packed into a composite input buffer::

        [original_input 0..max_orig] [ext_A 0..sA-1] [ext_B 0..sB-1] ...

    Returns
    -------
    segment_graph : IRGraph
        Neural-only graph with remapped sources.
    input_map : list[SegmentIOSlice]
        Describes how to assemble the composite input buffer from the
        global state buffer at runtime.
    """
    if weight_banks is None:
        weight_banks = {}
    node_ids = {n.id for n in nodes}

    for src in output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0 and src.node_id not in node_ids:
            raise ValueError(
                "Segment output_sources reference external node "
                f"(node_id={src.node_id})."
            )

    max_input_idx = -1
    for n in nodes:
        for src in n.input_sources.flatten():
            if isinstance(src, IRSource) and src.is_input():
                max_input_idx = max(max_input_idx, src.index)

    external_max_idx: dict[int, int] = {}
    for n in nodes:
        for src in n.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0 and src.node_id not in node_ids:
                external_max_idx[src.node_id] = max(
                    external_max_idx.get(src.node_id, 0), src.index
                )

    input_map: list[SegmentIOSlice] = []
    current_offset = (max_input_idx + 1) if max_input_idx >= 0 else 0

    if max_input_idx >= 0:
        input_map.append(SegmentIOSlice(node_id=-2, offset=0, size=max_input_idx + 1))

    offsets: dict[int, int] = {}
    for ext_id in sorted(external_max_idx.keys()):
        size = external_max_idx[ext_id] + 1
        offsets[ext_id] = current_offset
        input_map.append(SegmentIOSlice(node_id=ext_id, offset=current_offset, size=size))
        current_offset += size

    def remap_src(src: IRSource) -> IRSource:
        if src.node_id >= 0 and src.node_id not in node_ids:
            return IRSource(node_id=-2, index=offsets[src.node_id] + src.index)
        return src

    new_nodes: list[NeuralCore] = []
    for n in nodes:
        n2 = copy.deepcopy(n)
        flat = [remap_src(src) for src in n2.input_sources.flatten()]
        n2.input_sources = np.array(flat, dtype=object).reshape(n.input_sources.shape)
        new_nodes.append(n2)

    graph = IRGraph(
        nodes=new_nodes,
        output_sources=np.array(output_sources, dtype=object),
        weight_banks=weight_banks,
    )
    return graph, input_map


def _check_no_split_coalescing_groups(nodes: list) -> None:
    """Assert that no coalescing group is split at a segment boundary.

    Each coalescing group that has partial cores in this segment must also
    have its accumulator core in this segment.  A missing accumulator means
    a ComputeOp was inserted between a partial core and its accumulator,
    which would break the coalescing semantics.
    """
    group_roles: dict[int, list[str]] = {}
    for n in nodes:
        if isinstance(n, NeuralCore):
            gid = getattr(n, "coalescing_group_id", None)
            role = getattr(n, "coalescing_role", None)
            if gid is not None and role is not None:
                group_roles.setdefault(gid, []).append(role)

    for gid, roles in group_roles.items():
        if "master" in roles:
            continue # Unified wide core, no psums needed.

        if "accum" not in roles:
            raise ValueError(
                f"Coalescing group {gid} has partial cores in this neural segment "
                f"but its accumulator is missing (roles found: {roles}). "
                f"A ComputeOp must not be inserted between coalescing partial cores "
                f"and their accumulator."
            )


def _reindex_nodes(
    nodes: list[NeuralCore],
    reindex_maps: dict[int, dict[int, int]],
) -> list[NeuralCore]:
    """Deep-copy nodes and apply compaction reindex to their input_sources."""
    result = []
    for n in nodes:
        n2 = copy.deepcopy(n)
        _apply_reindex_to_ir_sources(n2.input_sources, reindex_maps)
        result.append(n2)
    return result


def _apply_reindex_to_ir_sources(
    sources: np.ndarray,
    reindex_maps: dict[int, dict[int, int]],
) -> None:
    """Apply compaction reindex maps to an array of IRSource objects in place.

    For each IRSource referencing a compacted core, updates ``.index``
    to the new (post-compaction) neuron index.  Sources referencing
    pruned neurons are replaced with off-sources.
    """
    for i, src in enumerate(sources.flatten()):
        if not isinstance(src, IRSource) or src.node_id < 0:
            continue
        remap = reindex_maps.get(src.node_id)
        if remap is None:
            continue  # core not in any compacted segment
        new_idx = remap.get(src.index)
        if new_idx is not None:
            sources.flat[i] = IRSource(node_id=src.node_id, index=new_idx)
        else:
            # Neuron was pruned — replace with off-source
            sources.flat[i] = IRSource(node_id=-1, index=0)


def _flush_neural_segment(
    *,
    current_neural: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    shared_pool: list[HardCore],
    weight_banks: dict,
    name: str,
    allow_neuron_splitting: bool = False,
    skip_coalescing_check: bool = False,
) -> tuple[HybridStage, dict[int, dict[int, int]]]:
    """Pack a neural segment using cores drawn from *shared_pool*.

    ``consumed_by`` is the pre-scanned map of *node_id → set of consumer
    node_ids* over the full IR graph.  It is used to determine which
    segment-internal nodes need to appear in the segment's output buffer
    (i.e. those consumed by any node *outside* this segment).

    When ``skip_coalescing_check`` is True, the coalescing group integrity
    check is skipped.  This is used by scheduled passes where coalescing
    fragments may be distributed across passes with the state buffer
    handling inter-pass data flow.

    Returns:
        (stage, reindex_maps) where reindex_maps is
        ``{core_id: {old_neuron_idx: new_neuron_idx}}`` from compaction.
        Callers must apply these maps to any external references that
        point into this segment's cores (e.g. final output_sources,
        ComputeOp input_sources).
    """
    segment_node_ids = {n.id for n in current_neural}

    if not skip_coalescing_check:
        _check_no_split_coalescing_groups(current_neural)

    output_nodes: list[NeuralCore] = []
    for n in current_neural:
        consumers = consumed_by.get(n.id, set())
        if any(c not in segment_node_ids for c in consumers):
            output_nodes.append(n)

    output_sources_list: list[IRSource] = []
    for n in output_nodes:
        for idx in range(n.get_output_count()):
            output_sources_list.append(IRSource(node_id=n.id, index=idx))

    if output_nodes and not output_sources_list:
        raise ValueError(
            "Segment has output_nodes but 0 output refs (all get_output_count() are 0). "
            "Output layer was over-pruned; at least one output neuron must remain."
        )

    output_sources = np.array(output_sources_list, dtype=object)

    seg_graph, input_map = _remap_external_sources_to_segment_inputs(
        nodes=current_neural,
        output_sources=output_sources,
        weight_banks=weight_banks,
    )
    soft = ir_graph_to_soft_core_mapping(seg_graph)

    # Compact soft cores: remove all-zero rows/columns and reindex spans so
    # hardware mapping shows only utilized structure (pruning reflected).
    reindex_maps = compact_soft_core_mapping(soft.cores, soft.output_sources)

    # Rebuild output_map from compacted output sizes
    output_map = []
    current_offset = 0
    output_core_ids = [n.id for n in output_nodes]
    core_by_id = {c.id: c for c in soft.cores}
    for nid in output_core_ids:
        core = core_by_id.get(nid)
        size = core.get_output_count() if core else 0
        output_map.append(SegmentIOSlice(node_id=nid, offset=current_offset, size=size))
        current_offset += size

    hard = HardCoreMapping(shared_pool)
    hard.map(soft, allow_neuron_splitting=allow_neuron_splitting)

    return HybridStage(
        kind="neural",
        name=name,
        hard_core_mapping=hard,
        input_map=input_map,
        output_map=output_map,
    ), reindex_maps


def _flush_scheduled_segment(
    *,
    current_neural: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    cores_config: Sequence[dict],
    weight_banks: dict,
    segment_index: int,
    segment_label: str,
    allow_neuron_splitting: bool = False,
) -> tuple[list[HybridStage], dict[int, dict[int, int]]]:
    """Flush a neural segment as one or more scheduled passes.

    Each pass gets a **fresh** hardware core pool (same physical cores
    reprogrammed).  Returns all stages and the accumulated reindex maps.

    Cores passed in must already be reindexed with any cross-segment
    compaction maps (the caller handles this).  This function only applies
    intra-segment inter-pass reindex (``seg_reindex_all``) to subsequent
    passes.

    If the initial partition produces a pass that fails to pack, the pass
    is automatically halved and retried until each sub-pass packs or
    contains only a single core.
    """
    from mimarsinan.mapping.schedule_partitioner import partition_segment_into_passes

    total_hw_cores = sum(int(ct["count"]) for ct in cores_config)
    max_hw_axons = max(int(ct["max_axons"]) for ct in cores_config) if cores_config else 0
    max_hw_neurons = max(int(ct["max_neurons"]) for ct in cores_config) if cores_config else 0

    passes = partition_segment_into_passes(
        current_neural, total_hw_cores,
        max_hw_axons=max_hw_axons,
        max_hw_neurons=max_hw_neurons,
        allow_coalescing=any(
            getattr(c, "coalescing_group_id", None) is not None for c in current_neural
        ),
        allow_splitting=allow_neuron_splitting,
    )

    stages: list[HybridStage] = []
    seg_reindex_all: dict[int, dict[int, int]] = {}

    def _flush_or_split(pass_cores, pass_idx):
        """Flush a pre-reindexed pass. On failure, split and retry.

        Cores must already be reindexed by the caller — this function
        does NOT apply any reindex maps itself.
        """
        fresh_pool = _make_available_hardware_cores(cores_config)
        try:
            stage, pass_reindex = _flush_neural_segment(
                current_neural=pass_cores,
                consumed_by=consumed_by,
                shared_pool=fresh_pool,
                weight_banks=weight_banks,
                name=f"{segment_label}_pass{pass_idx}",
                allow_neuron_splitting=allow_neuron_splitting,
                skip_coalescing_check=True,
            )
            stage.schedule_segment_index = segment_index
            stage.schedule_pass_index = pass_idx
            stages.append(stage)
            seg_reindex_all.update(pass_reindex)
        except RuntimeError:
            # Packing failed — split this pass in half and retry.
            if len(pass_cores) <= 1:
                raise  # Single core can't pack — truly infeasible
            mid = len(pass_cores) // 2
            snapshot = dict(seg_reindex_all)
            _flush_or_split(pass_cores[:mid], pass_idx)
            # Apply only the delta (first half's compaction) to second half
            delta = {k: v for k, v in seg_reindex_all.items() if k not in snapshot}
            second = _reindex_nodes(pass_cores[mid:], delta) if delta else pass_cores[mid:]
            _flush_or_split(second, pass_idx + 1000)

    for pass_idx, pass_cores in enumerate(passes):
        # Apply accumulated inter-pass reindex from previous passes in this
        # segment.  Cross-segment reindex was already applied by the caller.
        if seg_reindex_all:
            pass_cores = _reindex_nodes(pass_cores, seg_reindex_all)
        _flush_or_split(pass_cores, pass_idx)

    # Renumber pass indices sequentially after potential splits.
    for i, stage in enumerate(stages):
        if stage.schedule_segment_index == segment_index:
            stage.schedule_pass_index = i

    return stages, seg_reindex_all


def build_hybrid_hard_core_mapping(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    allow_neuron_splitting: bool = False,
    allow_scheduling: bool = False,
) -> HybridHardCoreMapping:
    """
    Compile a unified IRGraph into a HybridHardCoreMapping.

    Supports skip connections / residual paths through a state-buffer
    approach: each neural stage carries ``input_map`` / ``output_map``
    metadata so the runtime can maintain a ``Dict[int, Tensor]`` state
    buffer keyed by original IR node_id.

    When ``allow_scheduling`` is False (default), a **single** pool of
    hardware cores is allocated upfront and shared across all neural
    segments so the total core budget is respected.

    When ``allow_scheduling`` is True, each neural segment (or pass within
    a segment) gets a **fresh** hardware core pool — the same physical
    cores are reprogrammed between passes.  This allows mapping models
    that need more cores than available, trading latency for chip area.
    """

    consumed_by: dict[int, set[int]] = defaultdict(set)
    for node in ir_graph.nodes:
        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                consumed_by[src.node_id].add(node.id)
    for src in ir_graph.output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0:
            consumed_by[src.node_id].add(_FINAL_OUTPUT_SENTINEL)

    stages: list[HybridStage] = []

    # Collect compaction reindex maps from all segments so we can fix up
    # external references (final output_sources, ComputeOp input_sources).
    all_reindex_maps: dict[int, dict[int, int]] = {}

    if allow_scheduling:
        _build_scheduled(
            ir_graph=ir_graph,
            cores_config=cores_config,
            consumed_by=consumed_by,
            stages=stages,
            all_reindex_maps=all_reindex_maps,
            allow_neuron_splitting=allow_neuron_splitting,
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

    # Apply compaction reindex to final output_sources.
    output_sources = ir_graph.output_sources.copy()
    _apply_reindex_to_ir_sources(output_sources, all_reindex_maps)

    return HybridHardCoreMapping(
        stages=stages,
        output_sources=output_sources,
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
                # Apply accumulated reindex maps to nodes' input_sources so
                # references to previously-compacted cores use correct indices.
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

            # Apply compaction reindex to ComputeOp input_sources so they
            # reference the compacted neuron indices in the state buffer.
            op_copy = copy.deepcopy(node)
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


def _build_scheduled(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
    consumed_by: dict[int, set[int]],
    stages: list[HybridStage],
    all_reindex_maps: dict[int, dict[int, int]],
    allow_neuron_splitting: bool,
) -> None:
    """Scheduled compilation: fresh core pool per pass, segments may be split."""
    segment_index = 0

    current_neural: list[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current_neural.append(node)
            continue

        if isinstance(node, ComputeOp):
            if current_neural:
                if all_reindex_maps:
                    current_neural = _reindex_nodes(current_neural, all_reindex_maps)
                seg_stages, seg_reindex = _flush_scheduled_segment(
                    current_neural=current_neural,
                    consumed_by=consumed_by,
                    cores_config=cores_config,
                    weight_banks=ir_graph.weight_banks,
                    segment_index=segment_index,
                    segment_label=f"neural_segment_until:{node.name}",
                    allow_neuron_splitting=allow_neuron_splitting,
                )
                stages.extend(seg_stages)
                all_reindex_maps.update(seg_reindex)
                current_neural = []
                segment_index += 1

            op_copy = copy.deepcopy(node)
            _apply_reindex_to_ir_sources(op_copy.input_sources, all_reindex_maps)
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=op_copy))
            continue

        raise TypeError(f"Unknown IR node type in hybrid compilation: {type(node)}")

    if current_neural:
        if all_reindex_maps:
            current_neural = _reindex_nodes(current_neural, all_reindex_maps)
        seg_stages, seg_reindex = _flush_scheduled_segment(
            current_neural=current_neural,
            consumed_by=consumed_by,
            cores_config=cores_config,
            weight_banks=ir_graph.weight_banks,
            segment_index=segment_index,
            segment_label="neural_segment_final",
            allow_neuron_splitting=allow_neuron_splitting,
        )
        stages.extend(seg_stages)
        all_reindex_maps.update(seg_reindex)
