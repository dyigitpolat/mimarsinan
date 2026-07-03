from __future__ import annotations
from typing import Set
import numpy as np
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.pruning_apply import compact_hardware_bias_columns
from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
def _validate_outputs_remain(graph: IRGraph) -> None:
    if not graph.output_sources.size:
        return
    flat = graph.output_sources.flatten()
    if all(isinstance(s, IRSource) and s.node_id < 0 for s in flat):
        raise ValueError(
            "prune_ir_graph: all output_sources were rewired to pruned (node_id<0). "
            "At least one output neuron must remain; check initial pruning masks and propagation."
        )


def _compact_node(
    node: NeuralCore,
    *,
    pruned_rows: Set[int],
    pruned_cols: Set[int],
) -> None:
    """Drop pruned columns (and matching bias entries) then pruned rows.

    Liveness upstream guarantees a surviving core is never fully dead, so neither
    ``keep_cols`` nor ``keep_row_idx`` is ever empty; the asserts fail loud otherwise.
    """
    mat = node.core_matrix
    assert mat is not None, (
        f"_compact_node: NeuralCore id={node.id} ({node.name}) has no owned "
        "core_matrix; the caller must filter to owned-matrix cores."
    )
    if pruned_cols:
        keep_cols = [c for c in range(mat.shape[1]) if c not in pruned_cols]
        assert keep_cols, (
            f"_compact_node: NeuralCore id={node.id} ({node.name}) has every "
            "column pruned; the liveness pass should have removed it. This "
            "indicates a missing call to compute_liveness + remove_nodes "
            "in the surrounding pipeline."
        )
        mat = mat[:, keep_cols]
        node.core_matrix = mat
        node.hardware_bias = compact_hardware_bias_columns(
            node.hardware_bias, keep_cols
        )

    if pruned_rows:
        n_axons = mat.shape[0]
        keep_row_idx = np.fromiter(
            (r for r in range(n_axons) if r not in pruned_rows),
            dtype=np.int64,
            count=n_axons - len(pruned_rows),
        )
        flat_src = node.input_sources.flatten()
        if keep_row_idx.size > 0:
            node.core_matrix = mat[keep_row_idx, :]
            node.input_sources = flat_src[keep_row_idx]
        else:
            # BIAS_ONLY: all axons dead but bias-driven columns survive; collapse the row dim to a single OFF-source axon.
            node.core_matrix = np.zeros(
                (1, mat.shape[1]),
                dtype=mat.dtype,
            )
            node.input_sources = np.array(
                [IRSource(node_id=-1, index=0)], dtype=object,
            )


def _reset_post_compaction_masks(graph: IRGraph) -> None:
    """Move pre-compaction masks aside and re-derive same-shape post-compaction masks.

    Bank-backed cores keep the un-compacted masks (compacted physically at the soft-core
    mapping stage). A BIAS_ONLY core's single placeholder row is marked pruned.
    """
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        if getattr(node, "weight_bank_id", None) is not None:
            continue
        n_axons_new, n_neurons_new = node.core_matrix.shape
        if node.pruned_row_mask is not None and node.pruned_col_mask is not None:
            node.pre_pruning_row_mask = list(node.pruned_row_mask)
            node.pre_pruning_col_mask = list(node.pruned_col_mask)
        all_rows_were_pruned = bool(
            node.pre_pruning_row_mask is not None
            and len(node.pre_pruning_row_mask) > 0
            and all(node.pre_pruning_row_mask)
        )
        node.pruned_row_mask = [all_rows_were_pruned] * n_axons_new
        node.pruned_col_mask = [False] * n_neurons_new


def _attach_bank_metadata(
    graph: IRGraph,
    result: GlobalPruningResult,
    *,
    store_heatmap: bool,
) -> None:
    """Project bank-level pruned sets onto the per-node masks of bank-backed cores."""
    weight_banks = getattr(graph, "weight_banks", None) or {}
    for bank_id, bank in weight_banks.items():
        n_axons, n_neurons = bank.core_matrix.shape
        zero_rows = result.pruned_rows_per_bank.get(bank_id, set())
        zero_cols = result.pruned_cols_per_bank.get(bank_id, set())
        pruned_row_mask = [i in zero_rows for i in range(n_axons)]
        pruned_col_mask_full = [j in zero_cols for j in range(n_neurons)]
        for node in graph.nodes:
            if (
                not isinstance(node, NeuralCore)
                or getattr(node, "weight_bank_id", None) != bank_id
            ):
                continue
            if node.weight_row_slice is not None:
                start, end = node.weight_row_slice
                slice_mat = bank.core_matrix[:, start:end]
                node.pre_pruning_heatmap = (
                    np.asarray(slice_mat, dtype=np.float32) if store_heatmap else None
                )
                node.pruned_col_mask = pruned_col_mask_full[start:end]
            else:
                node.pre_pruning_heatmap = (
                    np.asarray(bank.core_matrix, dtype=np.float32) if store_heatmap else None
                )
                node.pruned_col_mask = pruned_col_mask_full
            node.pruned_row_mask = pruned_row_mask
