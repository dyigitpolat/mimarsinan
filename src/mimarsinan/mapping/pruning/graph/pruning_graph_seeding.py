"""Shared seeding + graph indexing for the global pruning modes (masked/closure/cascade)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AbstractSet, Dict, Mapping, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.liveness_transfer import (
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    DEFAULT_ELIMINATION_CONSTANT_FOLDING,
    ComputeOpTransferIndex,
    ConstantLattice,
    ELIMINATION_CONSTANT_FOLDING_FULL,
    domain_admits_nonzero_constants,
    effective_constant_folding,
)
from mimarsinan.mapping.pruning.graph.constant_folding import ConstantFoldState
from mimarsinan.mapping.pruning.graph.pruning_graph_types import (
    GlobalPruningResult,
    GraphIndex,
    _assert_index_matches,
    build_graph_index,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
    _cols_with_nonzero_bias,
    _resolve_node_matrix,
)


@dataclass
class GlobalPruningContext:
    """Seeded pruning state plus the graph indexes every mode shares."""

    graph: IRGraph
    zero_threshold: float
    neural_cores: list
    banks: Dict[int, WeightBank]
    exempt_rows: Dict[int, frozenset]
    exempt_cols: Dict[int, frozenset]
    consumer_axons: Dict[Tuple[int, int], list]
    model_output_neurons: Set[Tuple[int, int]]
    computeop_transfers: ComputeOpTransferIndex
    bank_consumers: Dict[int, Set[int]]
    bank_node_lookup: Dict[int, list]
    constants: ConstantFoldState = field(default_factory=ConstantFoldState)
    pruned_rows: Dict[int, Set[int]] = field(default_factory=dict)
    pruned_cols: Dict[int, Set[int]] = field(default_factory=dict)
    bank_pruned_rows: Dict[int, Set[int]] = field(default_factory=dict)
    bank_pruned_cols: Dict[int, Set[int]] = field(default_factory=dict)

    def base_node_matrix(self, node: NeuralCore) -> "np.ndarray | None":
        """The stored ``(axons, neurons)`` matrix (bank view resolved)."""
        return _resolve_node_matrix(node, self.banks)

    def node_matrix(self, node: NeuralCore) -> "np.ndarray | None":
        """The EFFECTIVE matrix every kernel must see: base + carrier delta."""
        base = _resolve_node_matrix(node, self.banks)
        if base is None:
            return None
        return self.constants.effective_matrix(node, base)

    def node_bias(self, node: NeuralCore) -> "np.ndarray | None":
        """The EFFECTIVE ``hardware_bias``: base + carrier delta."""
        return self.constants.effective_bias(node)

    def to_result(self, *, fixpoint_iterations: int) -> GlobalPruningResult:
        return GlobalPruningResult(
            pruned_rows_per_node=self.pruned_rows,
            pruned_cols_per_node=self.pruned_cols,
            pruned_rows_per_bank=self.bank_pruned_rows,
            pruned_cols_per_bank=self.bank_pruned_cols,
            fixpoint_iterations=fixpoint_iterations,
            constant_folds=self.constants,
        )


def build_global_pruning_context(
    graph: IRGraph,
    *,
    zero_threshold: float,
    initial_per_node: Mapping[int, Tuple[AbstractSet[int], AbstractSet[int]]] | None,
    initial_per_bank: Mapping[int, Tuple[AbstractSet[int], AbstractSet[int]]] | None,
    exempt_rows_per_node: Mapping[int, AbstractSet[int]] | None,
    exempt_cols_per_node: Mapping[int, AbstractSet[int]] | None,
    computeop_liveness_transfers: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    elimination_constant_folding: str = DEFAULT_ELIMINATION_CONSTANT_FOLDING,
    spiking_mode: str = "lif",
    probe_memo: "Dict[tuple, Dict[int, float]] | None" = None,
    graph_index: "GraphIndex | None" = None,
) -> GlobalPruningContext:
    """Index the graph and seed the pruned sets (explicit + off-source + value-based)."""
    neural_cores = [n for n in graph.nodes if isinstance(n, NeuralCore)]
    banks: Dict[int, WeightBank] = dict(getattr(graph, "weight_banks", {}) or {})
    index = graph_index if graph_index is not None else build_graph_index(
        graph, computeop_liveness_transfers=computeop_liveness_transfers,
    )
    _assert_index_matches(index, graph, computeop_liveness_transfers)

    exempt_rows = {n.id: frozenset(exempt_rows_per_node.get(n.id, set()))
                   for n in neural_cores} if exempt_rows_per_node else {}
    exempt_cols = {n.id: frozenset(exempt_cols_per_node.get(n.id, set()))
                   for n in neural_cores} if exempt_cols_per_node else {}
    if not exempt_rows:
        exempt_rows = {n.id: frozenset() for n in neural_cores}
    if not exempt_cols:
        exempt_cols = {n.id: frozenset() for n in neural_cores}

    ctx = GlobalPruningContext(
        graph=graph,
        zero_threshold=zero_threshold,
        neural_cores=neural_cores,
        banks=banks,
        exempt_rows=exempt_rows,
        exempt_cols=exempt_cols,
        consumer_axons=index.consumer_axons,
        model_output_neurons=index.model_output_neurons,
        computeop_transfers=index.computeop_transfers,
        constants=_build_constant_state(
            elimination_constant_folding=elimination_constant_folding,
            computeop_liveness_transfers=computeop_liveness_transfers,
            spiking_mode=spiking_mode,
            probe_memo=probe_memo,
        ),
        bank_consumers=index.bank_consumers,
        bank_node_lookup=index.bank_node_lookup,
        pruned_rows={n.id: set() for n in neural_cores},
        pruned_cols={n.id: set() for n in neural_cores},
        bank_pruned_rows={bid: set() for bid in banks},
        bank_pruned_cols={bid: set() for bid in banks},
    )

    if initial_per_node:
        for nid, (rows, cols) in initial_per_node.items():
            if nid not in ctx.pruned_rows:
                continue
            ctx.pruned_rows[nid] |= set(rows) - exempt_rows.get(nid, frozenset())
            ctx.pruned_cols[nid] |= set(cols) - exempt_cols.get(nid, frozenset())

    if initial_per_bank:
        for bid, (rows, cols) in initial_per_bank.items():
            if bid not in ctx.bank_pruned_rows:
                continue
            ctx.bank_pruned_rows[bid] |= set(rows)
            ctx.bank_pruned_cols[bid] |= set(cols)

    _seed_off_source_axons(neural_cores, ctx.pruned_rows, exempt_rows)
    _seed_value_based(
        neural_cores=neural_cores,
        banks=banks,
        zero_threshold=zero_threshold,
        pruned_rows=ctx.pruned_rows,
        pruned_cols=ctx.pruned_cols,
        bank_pruned_rows=ctx.bank_pruned_rows,
        bank_pruned_cols=ctx.bank_pruned_cols,
        exempt_rows=exempt_rows,
        exempt_cols=exempt_cols,
    )
    return ctx


def _build_constant_state(
    *,
    elimination_constant_folding: str,
    computeop_liveness_transfers: str,
    spiking_mode: str,
    probe_memo: "Dict[tuple, Dict[int, float]] | None" = None,
) -> ConstantFoldState:
    """The lattice, gated by the policy axis AND the chip-domain exactness gate."""
    policy = effective_constant_folding(
        policy=elimination_constant_folding,
        computeop_liveness_transfers=computeop_liveness_transfers,
    )
    return ConstantFoldState(
        enabled=policy == ELIMINATION_CONSTANT_FOLDING_FULL,
        lattice=ConstantLattice(
            admits_nonzero=domain_admits_nonzero_constants(spiking_mode)
        ),
        probe_memo=probe_memo if probe_memo is not None else {},
    )


def _seed_off_source_axons(
    neural_cores: list,
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
    neural_cores: list,
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
