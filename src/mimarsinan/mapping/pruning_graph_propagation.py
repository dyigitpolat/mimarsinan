"""Bidirectional, recursive cross-core pruning fixpoint over the unified IR.

This module provides the *global* propagation step. Within-matrix propagation
stays in :mod:`mimarsinan.mapping.pruning_propagation`; here we wire those
per-matrix closures together via the IR's source / consumer relationships:

- A NeuralCore axon ``i`` is dead when its source neuron ``(producer, j)`` is
  dead -- either ``IRSource.is_off()`` or in the producer's pruned columns.
- A NeuralCore neuron ``j`` is dead when no surviving consumer reads
  ``(producer, j)`` and it is not exempt (i.e. not a graph output).
- ComputeOp nodes are *barriers*: any neuron consumed by a ComputeOp is treated
  as having a live consumer (so it cannot be cross-core-pruned), and any axon
  fed by a ComputeOp output is treated as having a live source (so the
  ComputeOp side does not transmit deadness either).
- Exempt axons (model input data, ``IRSource.node_id == -2``) and exempt
  neurons (model output logits, in ``IRGraph.output_sources``) are never added
  to the pruned set.

For weight banks, we treat the bank matrix as a single shared object:
- ``bank.pruned_cols`` is the union of per-node pruned cols mapped to bank
  coordinates (each NeuralCore using the bank owns a contiguous slice).
- ``bank.pruned_rows`` is the *intersection* of per-node pruned rows (a row
  can only be physically pruned if every node using the bank has that axon
  dead). The result is then propagated back into per-node sets.

The fixpoint terminates because every iteration only adds elements to the
pruned sets, and the universe is bounded by ``sum(rows + cols)``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import AbstractSet, Dict, Mapping, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import (
    ComputeOp,
    IRGraph,
    IRSource,
    NeuralCore,
    WeightBank,
)
from mimarsinan.mapping.pruning_propagation import (
    compute_propagated_pruned_rows_cols,
)


@dataclass
class GlobalPruningResult:
    """Per-node and per-bank pruned row/column sets after global fixpoint."""

    pruned_rows_per_node: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols_per_node: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_rows_per_bank: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols_per_bank: Dict[int, Set[int]] = field(default_factory=dict)


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

    consumer_axons, neurons_with_persistent_consumer = _build_consumer_index(graph)
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
                neurons_with_persistent_consumer=neurons_with_persistent_consumer,
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
                neurons_with_persistent_consumer=neurons_with_persistent_consumer,
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
    """Index NeuralCore consumers and persistent-consumer markers per neuron.

    A neuron ``(producer_id, j)`` has a persistent consumer when it is
    referenced by ``output_sources`` or by any ComputeOp's ``input_sources``;
    persistence keeps the neuron alive against cross-core no-consumer pruning.
    """
    consumer_axons: Dict[Tuple[int, int], list[Tuple[int, int]]] = defaultdict(list)
    persistent: Set[Tuple[int, int]] = set()

    if graph.output_sources.size:
        for src in graph.output_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                persistent.add((src.node_id, src.index))

    for node in graph.nodes:
        if not hasattr(node, "input_sources"):
            continue
        for axon_idx, src in enumerate(node.input_sources.flatten()):
            if not isinstance(src, IRSource) or src.node_id < 0:
                continue
            key = (src.node_id, src.index)
            if isinstance(node, NeuralCore):
                consumer_axons[key].append((node.id, axon_idx))
            elif isinstance(node, ComputeOp):
                persistent.add(key)
            else:
                persistent.add(key)

    return consumer_axons, persistent


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


def _resolve_node_matrix(node: NeuralCore, banks: Mapping[int, WeightBank]) -> np.ndarray | None:
    """Return the effective ``(axons, neurons)`` matrix for a NeuralCore."""
    if node.core_matrix is not None:
        return node.core_matrix
    bid = getattr(node, "weight_bank_id", None)
    if bid is None or bid not in banks:
        return None
    bank = banks[bid]
    if node.weight_row_slice is not None:
        start, end = node.weight_row_slice
        return bank.core_matrix[:, start:end]
    return bank.core_matrix


def _cross_core_dead_axons(
    node: NeuralCore,
    pruned_cols: Mapping[int, AbstractSet[int]],
) -> Set[int]:
    """Axons whose source neuron is already dead (off or pruned)."""
    dead: Set[int] = set()
    for i, src in enumerate(node.input_sources.flatten()):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            dead.add(i)
            continue
        if src.node_id >= 0 and src.index in pruned_cols.get(src.node_id, frozenset()):
            dead.add(i)
    return dead


def _orphan_neurons(
    node_id: int,
    n_neurons: int,
    pruned_rows: Mapping[int, AbstractSet[int]],
    consumer_axons: Mapping[Tuple[int, int], list[Tuple[int, int]]],
    neurons_with_persistent_consumer: AbstractSet[Tuple[int, int]],
) -> Set[int]:
    """Neurons whose only NeuralCore consumers are dead axons (and no persistent consumer)."""
    dead: Set[int] = set()
    for j in range(n_neurons):
        key = (node_id, j)
        if key in neurons_with_persistent_consumer:
            continue
        consumers = consumer_axons.get(key, ())
        if not consumers:
            dead.add(j)
            continue
        if not any(
            axon_i not in pruned_rows.get(consumer_id, frozenset())
            for consumer_id, axon_i in consumers
        ):
            dead.add(j)
    return dead


def _refresh_node_pruning(
    *,
    node: NeuralCore,
    mat: np.ndarray,
    zero_threshold: float,
    pruned_rows: Dict[int, Set[int]],
    pruned_cols: Dict[int, Set[int]],
    consumer_axons: Mapping[Tuple[int, int], list[Tuple[int, int]]],
    neurons_with_persistent_consumer: AbstractSet[Tuple[int, int]],
    exempt_rows: Mapping[int, AbstractSet[int]],
    exempt_cols: Mapping[int, AbstractSet[int]],
) -> bool:
    """Rerun within-matrix propagation seeded with cross-core deadness.

    Returns True iff this iteration enlarged the node's pruned sets.
    """
    nid = node.id
    n_axons, n_neurons = mat.shape

    cross_rows = _cross_core_dead_axons(node, pruned_cols) - exempt_rows.get(
        nid, frozenset()
    )
    cross_cols = _orphan_neurons(
        nid,
        n_neurons,
        pruned_rows,
        consumer_axons,
        neurons_with_persistent_consumer,
    ) - exempt_cols.get(nid, frozenset())

    seed_rows = pruned_rows[nid] | cross_rows
    seed_cols = pruned_cols[nid] | cross_cols

    new_rows, new_cols = compute_propagated_pruned_rows_cols(
        mat,
        zero_threshold=zero_threshold,
        initial_zero_rows=seed_rows,
        initial_zero_cols=seed_cols,
        exempt_rows=exempt_rows.get(nid, frozenset()),
        exempt_cols=exempt_cols.get(nid, frozenset()),
        cols_with_implicit_source=_cols_with_nonzero_bias(
            getattr(node, "hardware_bias", None), n_neurons, zero_threshold
        ),
    )

    changed = new_rows != pruned_rows[nid] or new_cols != pruned_cols[nid]
    pruned_rows[nid] = new_rows
    pruned_cols[nid] = new_cols
    return changed


def _cols_with_nonzero_bias(
    hardware_bias, n_neurons: int, zero_threshold: float
) -> frozenset[int]:
    """Column indices whose ``hardware_bias`` magnitude is above threshold.

    A non-zero per-neuron bias keeps the corresponding column "alive" against
    within-matrix col-death propagation: even when every axon feeding the
    column is dead, the bias produces spikes on its own.
    """
    if hardware_bias is None:
        return frozenset()
    arr = np.asarray(hardware_bias)
    if arr.size != n_neurons:
        return frozenset()
    return frozenset(int(j) for j in np.flatnonzero(np.abs(arr) >= zero_threshold))


def _refresh_bank_pruning(
    *,
    bank: WeightBank,
    bank_id: int,
    zero_threshold: float,
    bank_nodes: list[NeuralCore],
    bank_consumers: AbstractSet[int],
    neurons_with_persistent_consumer: AbstractSet[Tuple[int, int]],
    pruned_rows: Dict[int, Set[int]],
    pruned_cols: Dict[int, Set[int]],
    bank_pruned_rows: Dict[int, Set[int]],
    bank_pruned_cols: Dict[int, Set[int]],
    exempt_rows: Mapping[int, AbstractSet[int]],
    exempt_cols: Mapping[int, AbstractSet[int]],
) -> bool:
    """Aggregate per-node bank views into bank-level pruned sets and project back.

    Bank rows are pruned only when *every* using node has the corresponding
    axon dead (we cannot drop a row another node still needs). Bank columns
    are the union over per-node pruned cols mapped to bank coords.
    """
    n_axons, n_neurons = bank.core_matrix.shape

    if bank_nodes:
        node_row_sets = [pruned_rows.get(n.id, set()) for n in bank_nodes]
        rows_intersection = set(node_row_sets[0])
        for s in node_row_sets[1:]:
            rows_intersection &= s
    else:
        rows_intersection = set()

    seed_cols = set(bank_pruned_cols[bank_id])
    bank_exempt_rows: Set[int] = set()
    bank_exempt_cols: Set[int] = set()
    for node in bank_nodes:
        nid = node.id
        if node.weight_row_slice is not None:
            start, _end = node.weight_row_slice
        else:
            start = 0
        for j_local in pruned_cols.get(nid, set()):
            seed_cols.add(start + j_local)
        bank_exempt_rows |= exempt_rows.get(nid, frozenset())
        for j_local in exempt_cols.get(nid, frozenset()):
            bank_exempt_cols.add(start + j_local)

    seed_rows = bank_pruned_rows[bank_id] | (rows_intersection - bank_exempt_rows)

    bank_bias_alive_cols = _cols_with_nonzero_bias(
        getattr(bank, "hardware_bias", None), n_neurons, zero_threshold
    )
    per_node_bias_alive_cols: Set[int] = set()
    for node in bank_nodes:
        if node.weight_row_slice is not None:
            start, _end = node.weight_row_slice
        else:
            start = 0
        node_bias = getattr(node, "hardware_bias", None)
        if node_bias is None:
            continue
        node_bias_arr = np.asarray(node_bias)
        if node_bias_arr.size == 0:
            continue
        for j in _cols_with_nonzero_bias(node_bias, node_bias_arr.size, zero_threshold):
            per_node_bias_alive_cols.add(start + j)

    new_rows, new_cols = compute_propagated_pruned_rows_cols(
        bank.core_matrix,
        zero_threshold=zero_threshold,
        initial_zero_rows=seed_rows,
        initial_zero_cols=seed_cols,
        exempt_rows=frozenset(bank_exempt_rows),
        exempt_cols=frozenset(bank_exempt_cols),
        cols_with_implicit_source=frozenset(
            bank_bias_alive_cols | per_node_bias_alive_cols
        ),
    )

    changed = (
        new_rows != bank_pruned_rows[bank_id]
        or new_cols != bank_pruned_cols[bank_id]
    )
    bank_pruned_rows[bank_id] = new_rows
    bank_pruned_cols[bank_id] = new_cols

    for node in bank_nodes:
        nid = node.id
        if node.weight_row_slice is not None:
            start, end = node.weight_row_slice
        else:
            start, end = 0, n_neurons

        ex_rows = exempt_rows.get(nid, frozenset())
        added_rows = (new_rows - pruned_rows.get(nid, set())) - ex_rows
        if added_rows:
            pruned_rows[nid] |= added_rows
            changed = True

        ex_cols = exempt_cols.get(nid, frozenset())
        view_cols = {j - start for j in new_cols if start <= j < end}
        added_cols = (view_cols - pruned_cols.get(nid, set())) - ex_cols
        if added_cols:
            pruned_cols[nid] |= added_cols
            changed = True

    # Suppress unused-arg lints for parameters retained for symmetry / future use
    _ = bank_consumers
    _ = neurons_with_persistent_consumer
    return changed
