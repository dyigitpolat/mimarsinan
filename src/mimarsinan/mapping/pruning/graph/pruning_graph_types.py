from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Set, Tuple

from collections import defaultdict

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.liveness_transfer import (
    build_computeop_transfer_index,
)
from mimarsinan.mapping.pruning.graph.constant_folding import ConstantFoldState


@dataclass
class GraphIndex:
    """The immutable structure every arm of one run shares.

    Pure functions of (graph, transfer policy): the analysis never mutates the
    graph, so one build serves masked/closure/cascade and the depth replay.
    ``graph_id``/``policy`` are carried so a context REFUSES a foreign or
    wrong-policy index instead of analyzing under it.
    """

    graph_id: int
    policy: str
    consumer_axons: Dict[Tuple[int, int], list]
    model_output_neurons: Set[Tuple[int, int]]
    computeop_transfers: Any
    bank_consumers: Dict[int, Set[int]]
    bank_node_lookup: Dict[int, list]

@dataclass
class GlobalPruningResult:
    """Per-node and per-bank pruned row/column sets after global fixpoint.

    ``fixpoint_iterations`` counts the global refresh sweeps executed
    (including the final quiescent one): 0 for the masked arm, 1 for the
    closure arm, and the actual sweep count for the cascade fixpoint.

    ``constant_folds`` [W4b-2] carries the constant lattice and the carrier
    deltas the fold plan accumulated. It is EMPTY (and its deltas are absent)
    whenever nothing folded, so the default path hands back the original
    arrays untouched.
    """

    pruned_rows_per_node: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols_per_node: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_rows_per_bank: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols_per_bank: Dict[int, Set[int]] = field(default_factory=dict)
    fixpoint_iterations: int = 0
    constant_folds: ConstantFoldState = field(default_factory=ConstantFoldState)


def build_graph_index(
    graph: IRGraph, *, computeop_liveness_transfers: str,
) -> GraphIndex:
    """The per-run immutable structure (transfer index dominates: 93 s at real scale)."""
    consumer_axons, model_output_neurons = _build_consumer_index(graph)
    neural_cores = [n for n in graph.nodes if isinstance(n, NeuralCore)]
    banks = dict(getattr(graph, "weight_banks", {}) or {})
    return GraphIndex(
        graph_id=id(graph),
        policy=str(computeop_liveness_transfers),
        consumer_axons=consumer_axons,
        model_output_neurons=model_output_neurons,
        computeop_transfers=build_computeop_transfer_index(
            graph, policy=computeop_liveness_transfers
        ),
        bank_consumers=_build_bank_consumer_map(neural_cores),
        bank_node_lookup={
            b: [n for n in neural_cores
                if getattr(n, "weight_bank_id", None) == b]
            for b in banks
        },
    )

def _assert_index_matches(index: GraphIndex, graph, policy: str) -> None:
    if index.graph_id != id(graph):
        raise ValueError(
            "build_global_pruning_context: the supplied graph_index was built "
            "for a different graph; sharing it would analyze foreign structure."
        )
    if index.policy != str(policy):
        raise ValueError(
            f"build_global_pruning_context: graph_index transfer policy "
            f"{index.policy!r} != requested {policy!r}."
        )

def _build_consumer_index(
    graph: IRGraph,
) -> Tuple[Dict[Tuple[int, int], list], Set[Tuple[int, int]]]:
    """Index NeuralCore axon consumers and model-output neuron markers.

    Model-output neurons (``output_sources``) are protected from orphan pruning.
    ComputeOp wiring is handled separately via the liveness-transfer index.
    """
    consumer_axons: Dict[Tuple[int, int], list] = defaultdict(list)
    model_output_neurons: Set[Tuple[int, int]] = set()

    if graph.output_sources.size:
        for src in graph.output_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                model_output_neurons.add((src.node_id, src.index))

    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or not hasattr(node, "input_sources"):
            continue
        for axon_idx, src in enumerate(node.input_sources.flatten()):
            if not isinstance(src, IRSource) or src.node_id < 0:
                continue
            consumer_axons[(src.node_id, src.index)].append((node.id, axon_idx))

    return consumer_axons, model_output_neurons

def _build_bank_consumer_map(neural_cores: list) -> Dict[int, Set[int]]:
    """For each weight bank, the set of NeuralCore ids that reference it."""
    out: Dict[int, Set[int]] = defaultdict(set)
    for n in neural_cores:
        bid = getattr(n, "weight_bank_id", None)
        if bid is not None:
            out[bid].add(n.id)
    return out
