from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore, ir_graph_to_soft_core_mapping
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping


@dataclass
class HybridStage:
    """
    A single stage in a hybrid runtime program.

    - "neural": A HardCoreMapping that can be executed on the chip runtime.
    - "compute": A ComputeOp that must be executed as a sync barrier (rate -> op -> respike).
    """

    kind: Literal["neural", "compute"]
    name: str
    hard_core_mapping: HardCoreMapping | None = None
    compute_op: ComputeOp | None = None


@dataclass
class HybridHardCoreMapping:
    """
    A deployable *hybrid* program representation:

    neural segment (HardCoreMapping) -> ComputeOp barrier -> neural segment -> ...

    Each neural segment can be packed/codegen'ed as a standalone chip program.
    ComputeOps represent host-side (or auxiliary) computation between chip runs.
    """

    stages: List[HybridStage]

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
) -> IRGraph:
    """
    Build a neural-only IRGraph for a segment.

    Any IRSource references to nodes outside this segment are rewritten to
    segment-local inputs (IRSource(node_id=-2, index=...)).

    Current supported assumption (by design, for clean staging):
    - External references must come from at most ONE upstream node_id (typically the
      immediate preceding ComputeOp).
    """

    node_ids = {n.id for n in nodes}

    external_node_ids: set[int] = set()
    for n in nodes:
        for src in n.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0 and src.node_id not in node_ids:
                external_node_ids.add(src.node_id)

    # Output sources must be satisfiable from this segment.
    for src in output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0 and src.node_id not in node_ids:
            raise ValueError(
                "Hybrid segment construction error: segment output_sources reference a node "
                f"outside the segment (node_id={src.node_id})."
            )

    if len(external_node_ids) > 1:
        # This implies a skip-connection / multi-input staging requirement. We can support this
        # later by introducing explicit segment-IO packing, but keep the runtime clean for now.
        raise NotImplementedError(
            "Hybrid staging currently supports external inputs from a single upstream node_id "
            f"(got {sorted(external_node_ids)})."
        )

    external_node_id = next(iter(external_node_ids), None)

    def remap_src(src: IRSource) -> IRSource:
        if src.node_id >= 0 and src.node_id not in node_ids:
            # Preserve index: this is critical for ComputeOp -> next neural segment wiring.
            return IRSource(node_id=-2, index=int(src.index))
        return src

    new_nodes: list[NeuralCore] = []
    for n in nodes:
        n2 = copy.deepcopy(n)
        flat = [remap_src(src) for src in n2.input_sources.flatten()]
        n2.input_sources = np.array(flat, dtype=object).reshape(n.input_sources.shape)
        new_nodes.append(n2)

    # If there was an external node id, sanity-check that all such references were remapped.
    if external_node_id is not None:
        for n in new_nodes:
            for src in n.input_sources.flatten():
                if isinstance(src, IRSource) and src.node_id == external_node_id:
                    raise AssertionError("Internal error: external source remapping incomplete.")

    return IRGraph(nodes=new_nodes, output_sources=np.array(output_sources, dtype=object))


def build_hybrid_hard_core_mapping(
    *,
    ir_graph: IRGraph,
    cores_config: Sequence[dict],
) -> HybridHardCoreMapping:
    """
    Compile a unified IRGraph into a HybridHardCoreMapping:
    - consecutive NeuralCore nodes are grouped into a segment and packed to HardCoreMapping
    - ComputeOp nodes become sync-barrier stages
    """

    stages: list[HybridStage] = []

    current_neural: list[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current_neural.append(node)
            continue

        if isinstance(node, ComputeOp):
            # Flush neural segment (if any): outputs are the ComputeOp inputs (barrier interface).
            if current_neural:
                seg_graph = _remap_external_sources_to_segment_inputs(
                    nodes=current_neural,
                    output_sources=node.input_sources,
                )
                soft = ir_graph_to_soft_core_mapping(seg_graph)
                hard = HardCoreMapping(_make_available_hardware_cores(cores_config))
                hard.map(soft)
                stages.append(
                    HybridStage(
                        kind="neural",
                        name=f"neural_segment_until:{node.name}",
                        hard_core_mapping=hard,
                    )
                )
                current_neural = []

            # Add compute barrier stage.
            stages.append(HybridStage(kind="compute", name=node.name, compute_op=copy.deepcopy(node)))
            continue

        raise TypeError(f"Unknown IR node type in hybrid compilation: {type(node)}")

    # Flush final neural segment: outputs are the IRGraph outputs.
    if current_neural:
        seg_graph = _remap_external_sources_to_segment_inputs(
            nodes=current_neural,
            output_sources=ir_graph.output_sources,
        )
        soft = ir_graph_to_soft_core_mapping(seg_graph)
        hard = HardCoreMapping(_make_available_hardware_cores(cores_config))
        hard.map(soft)
        stages.append(HybridStage(kind="neural", name="neural_segment_final", hard_core_mapping=hard))

    if not stages:
        raise ValueError("Cannot build HybridHardCoreMapping: IRGraph has no stages.")

    return HybridHardCoreMapping(stages=stages)


