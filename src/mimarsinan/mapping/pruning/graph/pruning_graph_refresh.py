from __future__ import annotations

from typing import AbstractSet, Dict, Mapping, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.graph.pruning_propagation import compute_propagated_pruned_rows_cols
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
    computeop_producer_map: Mapping[Tuple[int, int], Tuple[int, int]],
) -> Set[int]:
    """Axons whose source neuron is already dead (off, pruned, or via ComputeOp relay)."""
    dead: Set[int] = set()
    for i, src in enumerate(node.input_sources.flatten()):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            dead.add(i)
            continue
        upstream = computeop_producer_map.get((src.node_id, src.index))
        if upstream is not None:
            up_nid, up_col = upstream
            if up_col in pruned_cols.get(up_nid, frozenset()):
                dead.add(i)
        elif src.node_id >= 0 and src.index in pruned_cols.get(src.node_id, frozenset()):
            dead.add(i)
    return dead


def _orphan_neurons(
    node_id: int,
    n_neurons: int,
    pruned_rows: Mapping[int, AbstractSet[int]],
    consumer_axons: Mapping[Tuple[int, int], list[Tuple[int, int]]],
    model_output_neurons: AbstractSet[Tuple[int, int]],
    computeop_referenced_neurons: AbstractSet[Tuple[int, int]],
) -> Set[int]:
    """Neurons with no live NeuralCore consumers and no model/ComputeOp wiring."""
    dead: Set[int] = set()
    for j in range(n_neurons):
        key = (node_id, j)
        if key in model_output_neurons or key in computeop_referenced_neurons:
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
    model_output_neurons: AbstractSet[Tuple[int, int]],
    computeop_referenced_neurons: AbstractSet[Tuple[int, int]],
    computeop_producer_map: Mapping[Tuple[int, int], Tuple[int, int]],
    exempt_rows: Mapping[int, AbstractSet[int]],
    exempt_cols: Mapping[int, AbstractSet[int]],
) -> bool:
    """Rerun within-matrix propagation seeded with cross-core deadness.

    Returns True iff this iteration enlarged the node's pruned sets.
    """
    nid = node.id
    n_axons, n_neurons = mat.shape

    cross_rows = _cross_core_dead_axons(
        node, pruned_cols, computeop_producer_map
    ) - exempt_rows.get(nid, frozenset())
    cross_cols = _orphan_neurons(
        nid,
        n_neurons,
        pruned_rows,
        consumer_axons,
        model_output_neurons,
        computeop_referenced_neurons,
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
    model_output_neurons: AbstractSet[Tuple[int, int]],
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
    _ = model_output_neurons
    return changed
