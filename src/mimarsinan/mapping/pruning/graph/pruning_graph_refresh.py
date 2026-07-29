from __future__ import annotations

from typing import AbstractSet, Dict, Mapping, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CASCADE,
)
from mimarsinan.mapping.pruning.graph.pruning_propagation import compute_propagated_pruned_rows_cols
from mimarsinan.mapping.pruning.liveness_transfer import ComputeOpTransferIndex
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
    computeop_transfers: ComputeOpTransferIndex,
) -> Set[int]:
    """Axons whose source neuron is already dead (off, pruned, or via ComputeOp
    liveness transfer: an op output is dead when ALL of its transfer-mapped
    NeuralCore producers are dead)."""
    forward_producers = computeop_transfers.forward_producers
    dead: Set[int] = set()
    for i, src in enumerate(node.input_sources.flatten()):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            dead.add(i)
            continue
        producers = forward_producers.get((src.node_id, src.index))
        if producers is not None:
            if all(
                col in pruned_cols.get(nid, frozenset())
                for nid, col in producers
            ):
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
    computeop_transfers: ComputeOpTransferIndex,
) -> Set[int]:
    """Neurons with no live consumer — direct NeuralCore axons AND through-op
    (transfer-mapped) axons both dead. Transfer-protected ports (feeding an
    opaque op, or reaching a model output through ops) are never orphaned."""
    dead: Set[int] = set()
    protected = computeop_transfers.protected_ports
    effective = computeop_transfers.effective_consumers
    for j in range(n_neurons):
        key = (node_id, j)
        if key in model_output_neurons or key in protected:
            continue
        consumers = list(consumer_axons.get(key, ()))
        consumers.extend(effective.get(key, ()))
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
    hardware_bias: np.ndarray | None = None,
    zero_threshold: float,
    pruned_rows: Dict[int, Set[int]],
    pruned_cols: Dict[int, Set[int]],
    consumer_axons: Mapping[Tuple[int, int], list[Tuple[int, int]]],
    model_output_neurons: AbstractSet[Tuple[int, int]],
    computeop_transfers: ComputeOpTransferIndex,
    exempt_rows: Mapping[int, AbstractSet[int]],
    exempt_cols: Mapping[int, AbstractSet[int]],
    mode: str = ELIMINATION_PROPAGATION_CASCADE,
) -> bool:
    """Rerun within-matrix propagation seeded with cross-core deadness.

    ``mat`` / ``hardware_bias`` are the EFFECTIVE (post-constant-fold)
    structures; passing them explicitly keeps the kernel honest about the
    program it is reasoning over. ``hardware_bias=None`` falls back to the
    node's stored vector, which is exactly the pre-W4b-2 behaviour.

    Returns True iff this iteration enlarged the node's pruned sets.
    """
    nid = node.id
    if hardware_bias is None:
        hardware_bias = getattr(node, "hardware_bias", None)
    n_axons, n_neurons = mat.shape

    cross_rows = _cross_core_dead_axons(
        node, pruned_cols, computeop_transfers
    ) - exempt_rows.get(nid, frozenset())
    cross_cols = _orphan_neurons(
        nid,
        n_neurons,
        pruned_rows,
        consumer_axons,
        model_output_neurons,
        computeop_transfers,
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
            hardware_bias, n_neurons, zero_threshold
        ),
        mode=mode,
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
    mode: str = ELIMINATION_PROPAGATION_CASCADE,
) -> bool:
    """Aggregate per-node bank views into bank-level pruned sets and project back.

    Bank rows are pruned only when *every* using node has the corresponding
    axon dead (we cannot drop a row another node still needs). Bank columns
    follow the same rule per physical column: propagation-discovered deadness
    in ONE instance's view (starvation, orphaning) stays per-instance; the
    physical column dies only when every node whose ``weight_row_slice``
    covers it has its local view of the column dead. Explicit bank-level
    seeds (``bank_pruned_cols``) are shared by construction and pass through.
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
    covered = np.zeros(n_neurons, dtype=bool)
    dead_in_all_views = np.ones(n_neurons, dtype=bool)
    for node in bank_nodes:
        nid = node.id
        if node.weight_row_slice is not None:
            start, end = node.weight_row_slice
        else:
            start, end = 0, n_neurons
        node_dead = np.zeros(end - start, dtype=bool)
        for j_local in pruned_cols.get(nid, set()):
            if 0 <= j_local < end - start:
                node_dead[j_local] = True
        covered[start:end] = True
        dead_in_all_views[start:end] &= node_dead
        bank_exempt_rows |= exempt_rows.get(nid, frozenset())
        for j_local in exempt_cols.get(nid, frozenset()):
            bank_exempt_cols.add(start + j_local)
    # Union rule (columns): a physical column dies only if dead for ALL
    # instances whose slice covers it — one sharer's orphaned view must not
    # drop live signal of another sharer.
    seed_cols |= {int(c) for c in np.flatnonzero(covered & dead_in_all_views)}

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
        mode=mode,
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
