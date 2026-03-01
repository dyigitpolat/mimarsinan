from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore, ir_graph_to_soft_core_mapping
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping


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
        for _ in range(count):
            available_hardware_cores.append(HardCore(max_axons, max_neurons))
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


def _flush_neural_segment(
    *,
    current_neural: list[NeuralCore],
    consumed_by: dict[int, set[int]],
    shared_pool: list[HardCore],
    weight_banks: dict,
    name: str,
) -> HybridStage:
    """Pack a neural segment using cores drawn from *shared_pool*.

    ``consumed_by`` is the pre-scanned map of *node_id → set of consumer
    node_ids* over the full IR graph.  It is used to determine which
    segment-internal nodes need to appear in the segment's output buffer
    (i.e. those consumed by any node *outside* this segment).
    """
    segment_node_ids = {n.id for n in current_neural}

    output_nodes: list[NeuralCore] = []
    for n in current_neural:
        consumers = consumed_by.get(n.id, set())
        if any(c not in segment_node_ids for c in consumers):
            output_nodes.append(n)

    output_sources_list: list[IRSource] = []
    output_map: list[SegmentIOSlice] = []
    current_offset = 0
    for n in output_nodes:
        out_size = n.get_output_count()
        output_map.append(SegmentIOSlice(node_id=n.id, offset=current_offset, size=out_size))
        for idx in range(out_size):
            output_sources_list.append(IRSource(node_id=n.id, index=idx))
        current_offset += out_size

    output_sources = np.array(output_sources_list, dtype=object)

    seg_graph, input_map = _remap_external_sources_to_segment_inputs(
        nodes=current_neural,
        output_sources=output_sources,
        weight_banks=weight_banks,
    )
    soft = ir_graph_to_soft_core_mapping(seg_graph)

    hard = HardCoreMapping(shared_pool)
    hard.map(soft)

    return HybridStage(
        kind="neural",
        name=name,
        hard_core_mapping=hard,
        input_map=input_map,
        output_map=output_map,
    )


def build_hybrid_hard_core_mapping(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
) -> HybridHardCoreMapping:
    """
    Compile a unified IRGraph into a HybridHardCoreMapping.

    Supports skip connections / residual paths through a state-buffer
    approach: each neural stage carries ``input_map`` / ``output_map``
    metadata so the runtime can maintain a ``Dict[int, Tensor]`` state
    buffer keyed by original IR node_id.

    A **single** pool of hardware cores is allocated upfront and shared
    across all neural segments so the total core budget is respected.
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
    shared_pool: list[HardCore] = _make_available_hardware_cores(cores_config)

    current_neural: list[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current_neural.append(node)
            continue

        if isinstance(node, ComputeOp):
            if current_neural:
                stage = _flush_neural_segment(
                    current_neural=current_neural,
                    consumed_by=consumed_by,
                    shared_pool=shared_pool,
                    weight_banks=ir_graph.weight_banks,
                    name=f"neural_segment_until:{node.name}",
                )
                stages.append(stage)
                current_neural = []

            stages.append(HybridStage(kind="compute", name=node.name, compute_op=copy.deepcopy(node)))
            continue

        raise TypeError(f"Unknown IR node type in hybrid compilation: {type(node)}")

    if current_neural:
        stage = _flush_neural_segment(
            current_neural=current_neural,
            consumed_by=consumed_by,
            shared_pool=shared_pool,
            weight_banks=ir_graph.weight_banks,
            name="neural_segment_final",
        )
        stages.append(stage)

    if not stages:
        raise ValueError("Cannot build HybridHardCoreMapping: IRGraph has no stages.")

    return HybridHardCoreMapping(
        stages=stages,
        output_sources=ir_graph.output_sources.copy(),
    )
