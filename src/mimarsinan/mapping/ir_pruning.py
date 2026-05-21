"""Apply pruning to an :class:`IRGraph`: physically remove dead rows/columns
and rewire downstream sources.

The heavy lifting -- bidirectional, recursive cross-core propagation -- lives
in :mod:`mimarsinan.mapping.pruning_graph_propagation`. This module is the
pipeline-facing wrapper: it
1. collects model-level I/O exemptions
   (:func:`mimarsinan.mapping.ir_pruning_analysis.compute_graph_io_exemption`),
2. converts model-level seed masks into the solver's set form,
3. delegates to :func:`compute_global_pruned_sets` to find every dead row,
   column, and bank entry,
4. attaches GUI metadata (pre-pruning heatmaps and masks) for the heatmap
   renderer,
5. rewires consumer ``input_sources`` / ``output_sources``,
6. physically compacts owned ``NeuralCore.core_matrix`` and the bias.

Weight-bank backed cores keep the *un-compacted* bank matrix; the soft-core
mapping stage compacts them later. We only stash the per-node pruned-row and
pruned-col masks (sliced from the bank-level result) on the node.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.ir_pruning_analysis import compute_graph_io_exemption
from mimarsinan.mapping.pruning_apply import compact_hardware_bias_columns
from mimarsinan.mapping.pruning_graph_propagation import (
    GlobalPruningResult,
    compute_global_pruned_sets,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_initial_pruning_masks_from_model(model, ir_graph: IRGraph):
    """Collect per-node and per-bank pruning masks from the model layers.

    Returns ``(initial_pruned_per_node, initial_pruned_per_bank)`` where each
    map is keyed by the IR node / bank id and holds ``(row_mask, col_mask)``
    booleans aligned to that core's IR ``(axons, neurons)`` convention.
    ``True`` means *pruned*; bias rows are appended where present.
    """
    import torch

    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] = {}
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] = {}
    neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
    try:
        perceptrons = model.get_perceptrons()
    except Exception:
        return initial_pruned_per_node, initial_pruned_per_bank

    def _perceptron_masks(p):
        layer = getattr(p, "layer", None)
        if layer is None:
            return None, None
        prune_row = getattr(layer, "prune_row_mask", None)
        prune_col = getattr(layer, "prune_col_mask", None)
        if (
            prune_row is not None
            and isinstance(prune_row, torch.Tensor)
            and prune_row.dim() == 1
            and prune_col is not None
            and isinstance(prune_col, torch.Tensor)
            and prune_col.dim() == 1
        ):
            return prune_row.detach().cpu().numpy(), prune_col.detach().cpu().numpy()
        prune_mask = getattr(layer, "prune_mask", None)
        if prune_mask is None or not isinstance(prune_mask, torch.Tensor):
            return None, None
        pm = prune_mask.detach()
        prune_bias = getattr(layer, "prune_bias_mask", None)
        if prune_bias is not None:
            row_pruned = prune_bias.detach().cpu().numpy()
        else:
            row_pruned = pm.any(dim=1).cpu().numpy()
        col_pruned = pm.all(dim=0).cpu().numpy()
        return row_pruned, col_pruned

    for node in neural_cores:
        idx = getattr(node, "perceptron_index", None)
        if idx is None or idx < 0 or idx >= len(perceptrons):
            continue
        row_pruned, col_pruned = _perceptron_masks(perceptrons[idx])
        if row_pruned is None:
            continue
        out_slice = getattr(node, "perceptron_output_slice", None)
        in_slice = getattr(node, "perceptron_input_slice", None)
        if out_slice is not None:
            start, end = out_slice
            row_pruned = row_pruned[start:end]
        if in_slice is not None:
            start, end = in_slice
            col_pruned = col_pruned[start:end]
        in_f = len(col_pruned)
        out_f = len(row_pruned)
        try:
            mat = node.get_core_matrix(ir_graph)
            nr, nc = mat.shape
        except Exception:
            continue
        ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)]
        if nr > in_f:
            ir_row_mask.append(False)  # bias axon
        ir_row_mask = (ir_row_mask + [False] * nr)[:nr]
        ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
        ir_col_mask = (ir_col_mask + [False] * nc)[:nc]
        if len(ir_row_mask) == nr and len(ir_col_mask) == nc:
            initial_pruned_per_node[node.id] = (ir_row_mask, ir_col_mask)
        else:
            import logging
            logging.getLogger(__name__).debug(
                "get_initial_pruning_masks_from_model: skip node_id=%s shape mismatch "
                "nr=%s nc=%s len(ir_row_mask)=%s len(ir_col_mask)=%s",
                node.id, nr, nc, len(ir_row_mask), len(ir_col_mask),
            )

    for bank_id, bank in getattr(ir_graph, "weight_banks", {}).items():
        idx = getattr(bank, "perceptron_index", None)
        if idx is None or idx < 0 or idx >= len(perceptrons):
            continue
        row_pruned, col_pruned = _perceptron_masks(perceptrons[idx])
        if row_pruned is None:
            continue
        n_axons, n_neurons = bank.core_matrix.shape
        in_f = n_axons - 1
        out_f = n_neurons
        if len(col_pruned) != in_f or len(row_pruned) != out_f:
            continue
        ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)] + [False]
        ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
        if len(ir_row_mask) == n_axons and len(ir_col_mask) == n_neurons:
            initial_pruned_per_bank[bank_id] = (ir_row_mask, ir_col_mask)

    chip_perceptrons = [p for p in perceptrons if not getattr(p, "is_encoding_layer", False)]
    if (
        len(initial_pruned_per_node) == 0
        and len(initial_pruned_per_bank) == 0
        and len(neural_cores) == len(chip_perceptrons)
    ):
        for node, p in zip(neural_cores, chip_perceptrons):
            row_pruned, col_pruned = _perceptron_masks(p)
            if row_pruned is None:
                continue
            in_f, out_f = len(col_pruned), len(row_pruned)
            try:
                mat = node.get_core_matrix(ir_graph)
                nr, nc = mat.shape
            except Exception:
                continue
            if nr == in_f + 1 and nc == out_f:
                ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)] + [False]
                ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
            elif nr == in_f and nc == out_f:
                ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)]
                ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
            else:
                continue
            if len(ir_row_mask) == nr and len(ir_col_mask) == nc:
                initial_pruned_per_node[node.id] = (ir_row_mask, ir_col_mask)

    n_node = len(initial_pruned_per_node)
    n_bank = len(initial_pruned_per_bank)
    n_cores = len(neural_cores)
    n_perceptrons = len(perceptrons) if perceptrons else 0
    n_with_provenance = sum(
        1 for n in neural_cores if getattr(n, "perceptron_index", None) is not None
    )
    print(
        f"[Pruning] get_initial_pruning_masks_from_model: neural_cores={n_cores} "
        f"perceptrons={n_perceptrons} perceptron_index_set={n_with_provenance} "
        f"-> initial_pruned_per_node={n_node} initial_pruned_per_bank={n_bank}"
    )
    if neural_cores and perceptrons and n_node == 0 and n_bank == 0:
        p0 = perceptrons[0]
        layer = getattr(p0, "layer", None)
        has_1d = (
            layer is not None
            and getattr(layer, "prune_row_mask", None) is not None
            and getattr(layer, "prune_col_mask", None) is not None
        )
        has_2d = layer is not None and getattr(layer, "prune_mask", None) is not None
        print(
            f"[Pruning] No model masks applied. First perceptron layer: "
            f"prune_row_mask/prune_col_mask={has_1d} prune_mask={has_2d}. "
            "If both False, buffers were lost (H1/H5). If True, check provenance/shape (H2/H3/H4)."
        )

    return initial_pruned_per_node, initial_pruned_per_bank


def prune_ir_graph(
    ir_graph: IRGraph,
    zero_threshold: float = 1e-8,
    *,
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    store_heatmap: bool = False,
) -> IRGraph:
    """Prune and compact ``ir_graph`` in place; return the same instance.

    Pruning is bidirectional and recursive across NeuralCore boundaries.
    ComputeOp nodes act as barriers (their inputs and outputs always count as
    live). Model-level input data axons (``IRSource.node_id == -2``) and model
    output logits (entries in ``ir_graph.output_sources``) are never pruned.
    """
    if not ir_graph.nodes:
        return ir_graph

    graph = ir_graph

    exempt_rows, exempt_cols = _collect_exemptions(graph)
    seed_per_node, seed_per_bank = _collect_initial_seeds(
        graph, initial_pruned_per_node, initial_pruned_per_bank
    )

    result = compute_global_pruned_sets(
        graph,
        zero_threshold=zero_threshold,
        initial_per_node=seed_per_node,
        initial_per_bank=seed_per_bank,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
    )

    if not (initial_pruned_per_node or initial_pruned_per_bank):
        _log_value_based_summary(result)

    _attach_pre_compaction_metadata(graph, result, store_heatmap=store_heatmap)
    _rewire_sources(graph, result.pruned_cols_per_node)
    _validate_outputs_remain(graph)

    for node in graph.nodes:
        if isinstance(node, NeuralCore) and node.core_matrix is not None:
            _compact_node(
                node,
                pruned_rows=result.pruned_rows_per_node.get(node.id, set()),
                pruned_cols=result.pruned_cols_per_node.get(node.id, set()),
            )

    _reset_post_compaction_masks(graph)
    _attach_bank_metadata(graph, result, store_heatmap=store_heatmap)

    return graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _masks_to_sets(
    row_mask: Sequence[bool], col_mask: Sequence[bool]
) -> Tuple[Set[int], Set[int]]:
    """Convert boolean ``True = pruned`` masks to index sets."""
    return (
        {i for i, m in enumerate(row_mask) if m},
        {j for j, m in enumerate(col_mask) if m},
    )


def _collect_exemptions(
    graph: IRGraph,
) -> Tuple[Dict[int, frozenset], Dict[int, frozenset]]:
    """Per-node frozensets of rows/cols that must never be pruned."""
    in_buf, out_buf = compute_graph_io_exemption(graph)
    return (
        {nid: frozenset(s) for nid, s in in_buf.items()},
        {nid: frozenset(s) for nid, s in out_buf.items()},
    )


def _collect_initial_seeds(
    graph: IRGraph,
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None,
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None,
) -> Tuple[
    Dict[int, Tuple[Set[int], Set[int]]],
    Dict[int, Tuple[Set[int], Set[int]]],
]:
    """Convert model-supplied boolean masks into solver seed sets.

    Masks are clipped or zero-padded to the actual core / bank shape. Bias-row
    handling is implicit because callers pass ``False`` for bias.
    """
    seed_per_node: Dict[int, Tuple[Set[int], Set[int]]] = {}
    if initial_pruned_per_node:
        node_by_id = {n.id: n for n in graph.nodes if isinstance(n, NeuralCore)}
        for nid, (rows, cols) in initial_pruned_per_node.items():
            node = node_by_id.get(nid)
            if node is None or node.core_matrix is None:
                continue
            n_axons, n_neurons = node.core_matrix.shape
            row_mask = list(rows[:n_axons]) + [False] * max(0, n_axons - len(rows))
            col_mask = list(cols[:n_neurons]) + [False] * max(0, n_neurons - len(cols))
            seed_per_node[nid] = _masks_to_sets(row_mask, col_mask)

    seed_per_bank: Dict[int, Tuple[Set[int], Set[int]]] = {}
    banks = getattr(graph, "weight_banks", None) or {}
    if initial_pruned_per_bank:
        for bid, (rows, cols) in initial_pruned_per_bank.items():
            bank = banks.get(bid)
            if bank is None:
                continue
            n_axons, n_neurons = bank.core_matrix.shape
            if len(rows) != n_axons or len(cols) != n_neurons:
                continue
            seed_per_bank[bid] = _masks_to_sets(rows, cols)

    return seed_per_node, seed_per_bank


def _log_value_based_summary(result: GlobalPruningResult) -> None:
    """One-line summary when no model masks were supplied (zero-threshold prune)."""
    total_r = sum(len(s) for s in result.pruned_rows_per_node.values())
    total_c = sum(len(s) for s in result.pruned_cols_per_node.values())
    n_with_any = sum(
        1
        for nid in result.pruned_rows_per_node
        if result.pruned_rows_per_node[nid] or result.pruned_cols_per_node.get(nid)
    )
    print(
        f"[Pruning] prune_ir_graph: no model masks; value-based (zero_threshold): "
        f"nodes_with_pruned_rc={n_with_any} total_pruned_rows={total_r} total_pruned_cols={total_c}"
    )


def _attach_pre_compaction_metadata(
    graph: IRGraph,
    result: GlobalPruningResult,
    *,
    store_heatmap: bool,
) -> None:
    """Stash original matrix and prune masks for owned cores (used by GUI heatmaps)."""
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        mat = node.core_matrix
        n_axons, n_neurons = mat.shape
        zero_rows = result.pruned_rows_per_node.get(node.id, set())
        zero_cols = result.pruned_cols_per_node.get(node.id, set())
        node.pre_pruning_heatmap = (
            np.asarray(mat, dtype=np.float32) if store_heatmap else None
        )
        node.pruned_col_mask = [j in zero_cols for j in range(n_neurons)]
        node.pruned_row_mask = [i in zero_rows for i in range(n_axons)]


def _rewire_sources(
    graph: IRGraph,
    pruned_cols_per_node: Dict[int, Set[int]],
) -> None:
    """Replace pruned-neuron sources with off; reindex surviving neurons.

    Only owned-matrix NeuralCores are physically compacted in this stage, so
    only their producer-side neuron indices need rewiring. Bank-backed cores
    keep their neuron indexing untouched (the soft-core mapping stage compacts
    them later, and rewires consumers at that point).

    Mutates ``input_sources`` of every node and ``output_sources`` of the
    graph in place.
    """
    owned_node_ids = {
        n.id for n in graph.nodes
        if isinstance(n, NeuralCore) and n.core_matrix is not None
    }
    pruned_cols: Dict[int, Set[int]] = {
        nid: pruned_cols_per_node.get(nid, set()) for nid in owned_node_ids
    }
    reindex_maps: Dict[int, Dict[int, int]] = {}
    for node in graph.nodes:
        if node.id not in owned_node_ids:
            continue
        n_neurons = node.core_matrix.shape[1]
        zero_cols = pruned_cols[node.id]
        new_idx = 0
        remap: Dict[int, int] = {}
        for old_idx in range(n_neurons):
            if old_idx not in zero_cols:
                remap[old_idx] = new_idx
                new_idx += 1
        reindex_maps[node.id] = remap

    def _rewire(sources: np.ndarray) -> np.ndarray:
        flat = sources.flatten()
        for i, src in enumerate(flat):
            if not isinstance(src, IRSource) or src.node_id < 0:
                continue
            if src.node_id not in owned_node_ids:
                continue
            if src.index in pruned_cols[src.node_id]:
                flat[i] = IRSource(node_id=-1, index=0)
            elif src.index in reindex_maps[src.node_id]:
                flat[i] = IRSource(
                    node_id=src.node_id,
                    index=reindex_maps[src.node_id][src.index],
                )
            else:
                raise ValueError(
                    f"prune_ir_graph: source index {src.index} for node_id {src.node_id} "
                    "is neither pruned nor in reindex map; bookkeeping error upstream."
                )
        return flat.reshape(sources.shape)

    for node in graph.nodes:
        if hasattr(node, "input_sources"):
            node.input_sources = _rewire(node.input_sources)

    if graph.output_sources.size:
        graph.output_sources = _rewire(graph.output_sources)


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

    Maintains the contract that a NeuralCore always has at least one row and
    one column: when *every* row or column is pruned we keep a single zeroed
    placeholder so downstream code can still index into the matrix.
    """
    if pruned_cols:
        keep_cols = [c for c in range(node.core_matrix.shape[1]) if c not in pruned_cols]
        if keep_cols:
            node.core_matrix = node.core_matrix[:, keep_cols]
            node.hardware_bias = compact_hardware_bias_columns(
                node.hardware_bias, keep_cols
            )
        else:
            node.core_matrix = node.core_matrix[:, :1] * 0.0
            if node.hardware_bias is not None:
                node.hardware_bias = node.hardware_bias[:1] * 0.0

    if pruned_rows:
        n_axons = node.core_matrix.shape[0]
        keep_row_idx = np.fromiter(
            (r for r in range(n_axons) if r not in pruned_rows),
            dtype=np.int64,
            count=n_axons - len(pruned_rows),
        )
        flat_src = node.input_sources.flatten()
        if keep_row_idx.size > 0:
            node.core_matrix = node.core_matrix[keep_row_idx, :]
            node.input_sources = flat_src[keep_row_idx]
        else:
            node.core_matrix = node.core_matrix[:1, :] * 0.0
            node.input_sources = flat_src[:1]


def _reset_post_compaction_masks(graph: IRGraph) -> None:
    """Move pre-compaction masks aside and zero-fill new same-shape masks.

    Bank-backed cores keep the un-compacted masks (the soft-core mapping stage
    physically compacts banks later, and the masks must survive until then).

    Fully-pruned cores (every row and/or column dead) are compacted down to a
    ``(1, 1)`` zero placeholder by :func:`_compact_node`. The placeholder row
    and column do not represent live computation: they exist only so downstream
    code can still index into the matrix. Mark them as pruned in the
    post-compaction mask so the heatmap renderer overlays the prune lines and
    dead-math accounting (``ChipLatency``, soft-core mapper, GUI) treats them
    consistently with their pre-compaction status.
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
        all_cols_were_pruned = bool(
            node.pre_pruning_col_mask is not None
            and len(node.pre_pruning_col_mask) > 0
            and all(node.pre_pruning_col_mask)
        )
        node.pruned_row_mask = [all_rows_were_pruned] * n_axons_new
        node.pruned_col_mask = [all_cols_were_pruned] * n_neurons_new


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
