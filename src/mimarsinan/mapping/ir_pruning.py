"""Physically remove pruned rows/columns from IR NeuralCores and rewire sources."""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_pruning_analysis import (
    compute_segment_io_exemption,
    get_neural_segments,
)
from mimarsinan.mapping.pruning_propagation import compute_propagated_pruned_rows_cols


def get_initial_pruning_masks_from_model(model, ir_graph: IRGraph):
    """Collect per-node and per-bank pruning masks from the model (IR axon/neuron convention)."""
    import torch
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] = {}
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] = {}
    neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
    try:
        perceptrons = model.get_perceptrons()
    except Exception:
        return initial_pruned_per_node, initial_pruned_per_bank

    def _perceptron_masks(p):
        """Return (row_pruned, col_pruned) from perceptron layer masks."""
        layer = getattr(p, "layer", None)
        if layer is None:
            return None, None
        prune_row = getattr(layer, "prune_row_mask", None)
        prune_col = getattr(layer, "prune_col_mask", None)
        if prune_row is not None and isinstance(prune_row, torch.Tensor) and prune_row.dim() == 1:
            if prune_col is not None and isinstance(prune_col, torch.Tensor) and prune_col.dim() == 1:
                row_pruned = prune_row.detach().cpu().numpy()
                col_pruned = prune_col.detach().cpu().numpy()
                return row_pruned, col_pruned
        prune_mask = getattr(layer, "prune_mask", None)
        if prune_mask is None or not isinstance(prune_mask, torch.Tensor):
            return None, None
        pm = prune_mask.detach()
        out_f, in_f = pm.shape[0], pm.shape[1]
        prune_bias = getattr(layer, "prune_bias_mask", None)
        if prune_bias is not None:
            row_pruned = prune_bias.detach().cpu().numpy()  # (out_f,) True = pruned
        else:
            row_pruned = pm.any(dim=1).cpu().numpy()
        col_pruned = pm.all(dim=0).cpu().numpy()  # (in_f,) True = pruned (lossy when all rows pruned)
        return row_pruned, col_pruned

    for node in neural_cores:
        idx = getattr(node, "perceptron_index", None)
        if idx is None:
            continue
        if idx < 0 or idx >= len(perceptrons):
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
            ir_row_mask.append(False)  # bias
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
        if idx is None:
            continue
        if idx < 0 or idx >= len(perceptrons):
            continue
        row_pruned, col_pruned = _perceptron_masks(perceptrons[idx])
        if row_pruned is None:
            continue
        mat = bank.core_matrix
        n_axons, n_neurons = mat.shape
        in_f = n_axons - 1
        out_f = n_neurons
        if len(col_pruned) != in_f or len(row_pruned) != out_f:
            continue
        ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)] + [False]
        ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
        if len(ir_row_mask) == n_axons and len(ir_col_mask) == n_neurons:
            initial_pruned_per_bank[bank_id] = (ir_row_mask, ir_col_mask)

    chip_perceptrons = [p for p in perceptrons if not getattr(p, "is_encoding_layer", False)]
    if len(initial_pruned_per_node) == 0 and len(initial_pruned_per_bank) == 0 and len(neural_cores) == len(chip_perceptrons):
        for i, (node, p) in enumerate(zip(neural_cores, chip_perceptrons)):
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
    n_with_provenance = sum(1 for n in neural_cores if getattr(n, "perceptron_index", None) is not None)
    print(
        f"[Pruning] get_initial_pruning_masks_from_model: neural_cores={n_cores} perceptrons={n_perceptrons} "
        f"perceptron_index_set={n_with_provenance} -> initial_pruned_per_node={n_node} initial_pruned_per_bank={n_bank}"
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
            f"[Pruning] No model masks applied. First perceptron layer: prune_row_mask/prune_col_mask={has_1d} "
            f"prune_mask={has_2d}. If both False, buffers were lost (H1/H5). If True, check provenance/shape (H2/H3/H4)."
        )

    return initial_pruned_per_node, initial_pruned_per_bank


def _masks_to_sets(
    row_mask: Sequence[bool], col_mask: Sequence[bool]
) -> Tuple[Set[int], Set[int]]:
    """Convert boolean masks (True = pruned) to sets of indices."""
    zero_rows = {i for i, m in enumerate(row_mask) if m}
    zero_cols = {j for j, m in enumerate(col_mask) if m}
    return zero_rows, zero_cols


def _collect_bank_exemptions(
    graph: IRGraph,
    bank_id: int,
    input_buffer_rows: Dict[int, Set[int]],
    output_buffer_cols: Dict[int, Set[int]],
) -> Tuple[Set[int], Set[int]]:
    """Collect segment I/O exemption sets for all nodes sharing a weight bank."""
    exempt_rows: Set[int] = set()
    exempt_cols: Set[int] = set()
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or getattr(node, "weight_bank_id", None) != bank_id:
            continue
        exempt_rows |= input_buffer_rows.get(node.id, set())
        if node.weight_row_slice is not None:
            start, end = node.weight_row_slice
            for j in output_buffer_cols.get(node.id, set()):
                if j < end - start:
                    exempt_cols.add(start + j)
        else:
            exempt_cols |= output_buffer_cols.get(node.id, set())
    return exempt_rows, exempt_cols


def prune_ir_graph(
    ir_graph: IRGraph,
    zero_threshold: float = 1e-8,
    *,
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    store_heatmap: bool = False,
) -> IRGraph:
    """Prune and compact NeuralCores in place; optional pre-pruning heatmap for GUI."""
    if not ir_graph.nodes:
        return ir_graph

    graph = ir_graph

    read_sources: Set[Tuple[int, int]] = set()
    for src in graph.output_sources.flatten():
        node_id = getattr(src, "node_id", getattr(src, "core_", None))
        idx = getattr(src, "index", getattr(src, "neuron_", None))
        if node_id >= 0:
            read_sources.add((node_id, idx))
    for node in graph.nodes:
        if not hasattr(node, "input_sources"):
            continue
        for src in node.input_sources.flatten():
            node_id = getattr(src, "node_id", getattr(src, "core_", None))
            idx = getattr(src, "index", getattr(src, "neuron_", None))
            if node_id >= 0:
                read_sources.add((node_id, idx))

    prephase_dead_rows: Dict[int, Set[int]] = {}
    prephase_dead_cols: Dict[int, Set[int]] = {}
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        flat_src = node.input_sources.flatten()
        dead_r = {
            i for i, src in enumerate(flat_src)
            if getattr(src, "node_id", getattr(src, "core_", None)) == -1
        }
        n_neurons = node.core_matrix.shape[1]
        dead_c = {j for j in range(n_neurons) if (node.id, j) not in read_sources}
        if dead_r:
            prephase_dead_rows[node.id] = dead_r
        if dead_c:
            prephase_dead_cols[node.id] = dead_c

    input_buffer_rows, output_buffer_cols = compute_segment_io_exemption(graph)

    reindex_maps: Dict[int, Dict[int, int]] = {}
    pruned_neurons: Dict[int, Set[int]] = {}
    pruned_rows: Dict[int, Set[int]] = {}
    init_node = (initial_pruned_per_node or {})

    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        mat = node.core_matrix  # (axons, neurons)
        n_axons, n_neurons = mat.shape
        exempt_r = frozenset(input_buffer_rows.get(node.id, set()))
        exempt_c = frozenset(output_buffer_cols.get(node.id, set()))

        init = init_node.get(node.id)
        if init is not None:
            init_row = list(init[0][:n_axons]) + [False] * max(0, n_axons - len(init[0]))
            init_col = list(init[1][:n_neurons]) + [False] * max(0, n_neurons - len(init[1]))
            if len(init_row) == n_axons and len(init_col) == n_neurons:
                init_rows, init_cols = _masks_to_sets(init_row, init_col)
            else:
                init_rows, init_cols = None, None
        else:
            init_rows, init_cols = None, None

        pre_r = prephase_dead_rows.get(node.id, set())
        pre_c = prephase_dead_cols.get(node.id, set())
        if init_rows is not None:
            init_rows = init_rows | pre_r
            init_cols = init_cols | pre_c
        else:
            init_rows = pre_r if pre_r else None
            init_cols = pre_c if pre_c else None

        zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
            mat,
            zero_threshold=zero_threshold,
            initial_zero_rows=init_rows,
            initial_zero_cols=init_cols,
            exempt_rows=exempt_r,
            exempt_cols=exempt_c,
        )

        pruned_neurons[node.id] = zero_cols
        pruned_rows[node.id] = zero_rows

        new_idx = 0
        remap = {}
        for old_idx in range(n_neurons):
            if old_idx not in zero_cols:
                remap[old_idx] = new_idx
                new_idx += 1
        reindex_maps[node.id] = remap

    if not (initial_pruned_per_node or initial_pruned_per_bank):
        total_r = sum(len(s) for s in pruned_rows.values())
        total_c = sum(len(s) for s in pruned_neurons.values())
        n_with_any = sum(1 for nid in pruned_rows if pruned_rows[nid] or pruned_neurons.get(nid, set()))
        print(
            f"[Pruning] prune_ir_graph: no model masks; value-based (zero_threshold): "
            f"nodes_with_pruned_rc={n_with_any} total_pruned_rows={total_r} total_pruned_cols={total_c}"
        )

    def _rewire_sources(sources: np.ndarray) -> np.ndarray:
        """Rewire source references: pruned neurons -> off, others -> reindexed."""
        flat = sources.flatten()
        for i, src in enumerate(flat):
            if not isinstance(src, IRSource):
                continue
            if src.node_id < 0:
                continue
            if src.node_id in pruned_neurons:
                if src.index in pruned_neurons[src.node_id]:
                    flat[i] = IRSource(node_id=-1, index=0)
                elif src.node_id in reindex_maps and src.index in reindex_maps[src.node_id]:
                    flat[i] = IRSource(node_id=src.node_id, index=reindex_maps[src.node_id][src.index])
                elif src.node_id in reindex_maps:
                    raise ValueError(
                        f"prune_ir_graph: source index {src.index} for node_id {src.node_id} is neither "
                        "pruned nor in reindex map; bookkeeping error upstream."
                    )
        return flat.reshape(sources.shape)

    for node in graph.nodes:
        node.input_sources = _rewire_sources(node.input_sources)

    if graph.output_sources.size:
        graph.output_sources = _rewire_sources(graph.output_sources)

    if graph.output_sources.size:
        flat = graph.output_sources.flatten()
        if all(
            isinstance(s, IRSource) and s.node_id < 0
            for s in flat
        ):
            raise ValueError(
                "prune_ir_graph: all output_sources were rewired to pruned (node_id<0). "
                "At least one output neuron must remain; check initial pruning masks and propagation."
            )

    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        mat = node.core_matrix
        n_axons, n_neurons = mat.shape
        zero_cols = pruned_neurons.get(node.id, set())
        zero_rows_set = pruned_rows.get(node.id, set())
        if store_heatmap:
            node.pre_pruning_heatmap = np.asarray(mat, dtype=np.float32)
        else:
            node.pre_pruning_heatmap = None
        node.pruned_col_mask = [j in zero_cols for j in range(n_neurons)]
        node.pruned_row_mask = [r in zero_rows_set for r in range(n_axons)]

    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        zero_cols = pruned_neurons.get(node.id, set())
        if zero_cols:
            keep_cols = [c for c in range(node.core_matrix.shape[1]) if c not in zero_cols]
            if keep_cols:
                node.core_matrix = node.core_matrix[:, keep_cols]
                if node.hardware_bias is not None:
                    node.hardware_bias = node.hardware_bias[keep_cols]
            else:
                node.core_matrix = node.core_matrix[:, :1] * 0.0
                if node.hardware_bias is not None:
                    node.hardware_bias = node.hardware_bias[:1] * 0.0

    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        mat = node.core_matrix
        n_axons = mat.shape[0]
        zero_rows_set = pruned_rows.get(node.id, set())
        if not zero_rows_set:
            continue

        keep_row_idx = np.fromiter(
            (r for r in range(n_axons) if r not in zero_rows_set),
            dtype=np.int64,
            count=n_axons - len(zero_rows_set),
        )

        flat_src = node.input_sources.flatten()
        if keep_row_idx.size > 0:
            node.core_matrix = mat[keep_row_idx, :]
            node.input_sources = flat_src[keep_row_idx]
        else:
            node.core_matrix = mat[:1, :] * 0.0
            node.input_sources = flat_src[:1]

    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        if getattr(node, "weight_bank_id", None) is not None:
            continue
        mat = node.core_matrix
        n_axons_new, n_neurons_new = mat.shape[0], mat.shape[1]
        if node.pruned_row_mask is not None and node.pruned_col_mask is not None:
            node.pre_pruning_row_mask = list(node.pruned_row_mask)
            node.pre_pruning_col_mask = list(node.pruned_col_mask)
        node.pruned_row_mask = [False] * n_axons_new
        node.pruned_col_mask = [False] * n_neurons_new

    weight_banks = getattr(graph, "weight_banks", None) or {}
    init_bank = (initial_pruned_per_bank or {})
    for bank_id, bank in weight_banks.items():
        mat = bank.core_matrix
        n_axons, n_neurons = mat.shape
        init = init_bank.get(bank_id)
        bank_exempt_rows, bank_exempt_cols = _collect_bank_exemptions(
            graph, bank_id, input_buffer_rows, output_buffer_cols
        )
        exempt_r = frozenset(bank_exempt_rows)
        exempt_c = frozenset(bank_exempt_cols)
        if init is not None and len(init[0]) == n_axons and len(init[1]) == n_neurons:
            zero_rows, zero_cols = _masks_to_sets(init[0], init[1])
            zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
                mat,
                zero_threshold=zero_threshold,
                initial_zero_rows=zero_rows,
                initial_zero_cols=zero_cols,
                exempt_rows=exempt_r,
                exempt_cols=exempt_c,
            )
        else:
            zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
                mat,
                zero_threshold=zero_threshold,
                exempt_rows=exempt_r,
                exempt_cols=exempt_c,
            )
        pruned_row_mask = [i in zero_rows for i in range(n_axons)]
        pruned_col_mask_full = [j in zero_cols for j in range(n_neurons)]
        for node in graph.nodes:
            if not isinstance(node, NeuralCore) or getattr(node, "weight_bank_id", None) != bank_id:
                continue
            if node.weight_row_slice is not None:
                start, end = node.weight_row_slice
                if store_heatmap:
                    node.pre_pruning_heatmap = np.asarray(
                        bank.core_matrix[:, start:end], dtype=np.float32
                    )
                else:
                    node.pre_pruning_heatmap = None
                node.pruned_col_mask = pruned_col_mask_full[start:end]
            else:
                if store_heatmap:
                    node.pre_pruning_heatmap = np.asarray(
                        bank.core_matrix, dtype=np.float32
                    )
                else:
                    node.pre_pruning_heatmap = None
                node.pruned_col_mask = pruned_col_mask_full
            node.pruned_row_mask = pruned_row_mask

    return graph
