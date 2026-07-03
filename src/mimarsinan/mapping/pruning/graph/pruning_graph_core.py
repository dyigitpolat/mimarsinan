from __future__ import annotations
from collections import defaultdict
from typing import AbstractSet, Dict, Mapping, Set, Tuple
import numpy as np
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.boundary_policy import (
    build_computeop_producer_map,
    build_computeop_referenced_neurons,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
    _cols_with_nonzero_bias,
    _refresh_bank_pruning,
    _refresh_node_pruning,
    _resolve_node_matrix,
)
def compute_global_pruned_sets(
    graph: IRGraph,
    *,
    zero_threshold: float = 1e-8,
    initial_per_node: Mapping[int, Tuple[AbstractSet[int], AbstractSet[int]]] | None = None,
    initial_per_bank: Mapping[int, Tuple[AbstractSet[int], AbstractSet[int]]] | None = None,
    exempt_rows_per_node: Mapping[int, AbstractSet[int]] | None = None,
    exempt_cols_per_node: Mapping[int, AbstractSet[int]] | None = None,
) -> GlobalPruningResult:
    """Run the bidirectional, recursive cross-core pruning fixpoint.

    Args:
        graph: Unified IR graph. Pruning runs pre-segmentation: ``IRSource``
            with ``node_id == -2`` denotes model input data, and entries in
            ``graph.output_sources`` denote model output logits.
        zero_threshold: Sum-of-abs threshold for value-based per-matrix init.
        initial_per_node: Optional ``{node_id: (rows, cols)}`` seed sets;
            indices that are also exempt are silently dropped.
        initial_per_bank: Optional ``{bank_id: (rows, cols)}`` seed sets in the
            bank's own coordinate system.
        exempt_rows_per_node: Per-node row indices that must never be pruned.
        exempt_cols_per_node: Per-node column indices that must never be pruned.
    """
    neural_cores = [n for n in graph.nodes if isinstance(n, NeuralCore)]
    banks: Dict[int, WeightBank] = dict(getattr(graph, "weight_banks", {}) or {})

    if not neural_cores and not banks:
        return GlobalPruningResult()

    exempt_rows = {n.id: frozenset(exempt_rows_per_node.get(n.id, set()))
                   for n in neural_cores} if exempt_rows_per_node else {}
    exempt_cols = {n.id: frozenset(exempt_cols_per_node.get(n.id, set()))
                   for n in neural_cores} if exempt_cols_per_node else {}
    if not exempt_rows:
        exempt_rows = {n.id: frozenset() for n in neural_cores}
    if not exempt_cols:
        exempt_cols = {n.id: frozenset() for n in neural_cores}

    consumer_axons, model_output_neurons = _build_consumer_index(graph)
    computeop_referenced = build_computeop_referenced_neurons(graph)
    computeop_producer_map = build_computeop_producer_map(graph)
    bank_consumers = _build_bank_consumer_map(neural_cores)

    pruned_rows: Dict[int, Set[int]] = {n.id: set() for n in neural_cores}
    pruned_cols: Dict[int, Set[int]] = {n.id: set() for n in neural_cores}
    bank_pruned_rows: Dict[int, Set[int]] = {bid: set() for bid in banks}
    bank_pruned_cols: Dict[int, Set[int]] = {bid: set() for bid in banks}

    if initial_per_node:
        for nid, (rows, cols) in initial_per_node.items():
            if nid not in pruned_rows:
                continue
            pruned_rows[nid] |= set(rows) - exempt_rows.get(nid, frozenset())
            pruned_cols[nid] |= set(cols) - exempt_cols.get(nid, frozenset())

    if initial_per_bank:
        for bid, (rows, cols) in initial_per_bank.items():
            if bid not in bank_pruned_rows:
                continue
            bank_pruned_rows[bid] |= set(rows)
            bank_pruned_cols[bid] |= set(cols)

    _seed_off_source_axons(neural_cores, pruned_rows, exempt_rows)
    _seed_value_based(
        neural_cores=neural_cores,
        banks=banks,
        zero_threshold=zero_threshold,
        pruned_rows=pruned_rows,
        pruned_cols=pruned_cols,
        bank_pruned_rows=bank_pruned_rows,
        bank_pruned_cols=bank_pruned_cols,
        exempt_rows=exempt_rows,
        exempt_cols=exempt_cols,
    )

    bank_node_lookup = {
        b: [n for n in neural_cores if getattr(n, "weight_bank_id", None) == b]
        for b in banks
    }

    while True:
        changed = False
        for node in neural_cores:
            mat = _resolve_node_matrix(node, banks)
            if mat is None:
                continue
            if _refresh_node_pruning(
                node=node,
                mat=mat,
                zero_threshold=zero_threshold,
                pruned_rows=pruned_rows,
                pruned_cols=pruned_cols,
                consumer_axons=consumer_axons,
                model_output_neurons=model_output_neurons,
                computeop_referenced_neurons=computeop_referenced,
                computeop_producer_map=computeop_producer_map,
                exempt_rows=exempt_rows,
                exempt_cols=exempt_cols,
            ):
                changed = True

        for bank_id, bank in banks.items():
            if _refresh_bank_pruning(
                bank=bank,
                bank_id=bank_id,
                zero_threshold=zero_threshold,
                bank_nodes=bank_node_lookup[bank_id],
                bank_consumers=bank_consumers.get(bank_id, set()),
                model_output_neurons=model_output_neurons,
                pruned_rows=pruned_rows,
                pruned_cols=pruned_cols,
                bank_pruned_rows=bank_pruned_rows,
                bank_pruned_cols=bank_pruned_cols,
                exempt_rows=exempt_rows,
                exempt_cols=exempt_cols,
            ):
                changed = True

        if not changed:
            break

    return GlobalPruningResult(
        pruned_rows_per_node=pruned_rows,
        pruned_cols_per_node=pruned_cols,
        pruned_rows_per_bank=bank_pruned_rows,
        pruned_cols_per_bank=bank_pruned_cols,
    )


def _build_consumer_index(
    graph: IRGraph,
) -> Tuple[Dict[Tuple[int, int], list[Tuple[int, int]]], Set[Tuple[int, int]]]:
    """Index NeuralCore axon consumers and model-output neuron markers.

    Model-output neurons (``output_sources``) are protected from orphan pruning.
    ComputeOp wiring is handled separately via ``computeop_referenced_neurons``.
    """
    consumer_axons: Dict[Tuple[int, int], list[Tuple[int, int]]] = defaultdict(list)
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


def _build_bank_consumer_map(
    neural_cores: list[NeuralCore],
) -> Dict[int, Set[int]]:
    """For each weight bank, the set of NeuralCore ids that reference it."""
    out: Dict[int, Set[int]] = defaultdict(set)
    for n in neural_cores:
        bid = getattr(n, "weight_bank_id", None)
        if bid is not None:
            out[bid].add(n.id)
    return out


def _seed_off_source_axons(
    neural_cores: list[NeuralCore],
    pruned_rows: Dict[int, Set[int]],
    exempt_rows: Mapping[int, AbstractSet[int]],
) -> None:
    """Mark every axon whose ``IRSource.is_off()`` as initially pruned."""
    for node in neural_cores:
        if not hasattr(node, "input_sources"):
            continue
        exempt = exempt_rows.get(node.id, frozenset())
        flat = node.input_sources.flatten()
        for i, src in enumerate(flat):
            if (
                isinstance(src, IRSource)
                and src.is_off()
                and i not in exempt
            ):
                pruned_rows[node.id].add(i)


def _seed_value_based(
    *,
    neural_cores: list[NeuralCore],
    banks: Mapping[int, WeightBank],
    zero_threshold: float,
    pruned_rows: Dict[int, Set[int]],
    pruned_cols: Dict[int, Set[int]],
    bank_pruned_rows: Dict[int, Set[int]],
    bank_pruned_cols: Dict[int, Set[int]],
    exempt_rows: Mapping[int, AbstractSet[int]],
    exempt_cols: Mapping[int, AbstractSet[int]],
) -> None:
    """Union value-based dead rows/cols into the per-node and per-bank pruned sets.

    A row/column whose every weight is below ``zero_threshold`` cannot
    contribute to any downstream computation. It is pruned regardless of
    whether the caller supplied an explicit model mask: the model mask states
    *additional* deadness that the runtime weights may not yet reflect, but
    weights that are already zero are dead by themselves. Exemptions
    (``exempt_rows`` / ``exempt_cols``) and ``hardware_bias``-alive columns
    are still respected.
    """
    for node in neural_cores:
        mat = _resolve_node_matrix(node, banks)
        if mat is None:
            continue
        abs_mat = np.abs(np.asarray(mat))
        row_sum = abs_mat.sum(axis=1)
        col_sum = abs_mat.sum(axis=0)
        ex_r = exempt_rows.get(node.id, frozenset())
        ex_c = exempt_cols.get(node.id, frozenset())
        bias_alive_cols = _cols_with_nonzero_bias(
            getattr(node, "hardware_bias", None), mat.shape[1], zero_threshold
        )
        for i in np.flatnonzero(row_sum < zero_threshold):
            i = int(i)
            if i not in ex_r:
                pruned_rows[node.id].add(i)
        for j in np.flatnonzero(col_sum < zero_threshold):
            j = int(j)
            if j in ex_c or j in bias_alive_cols:
                continue
            pruned_cols[node.id].add(j)

    for bank_id, bank in banks.items():
        abs_mat = np.abs(np.asarray(bank.core_matrix))
        bias_alive_cols = _cols_with_nonzero_bias(
            getattr(bank, "hardware_bias", None),
            bank.core_matrix.shape[1],
            zero_threshold,
        )
        for i in np.flatnonzero(abs_mat.sum(axis=1) < zero_threshold):
            bank_pruned_rows[bank_id].add(int(i))
        for j in np.flatnonzero(abs_mat.sum(axis=0) < zero_threshold):
            j = int(j)
            if j in bias_alive_cols:
                continue
            bank_pruned_cols[bank_id].add(j)


