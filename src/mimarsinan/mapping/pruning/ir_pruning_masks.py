from __future__ import annotations
import logging
from typing import Dict, Sequence, Tuple
import torch
from mimarsinan.mapping.ir import IRGraph, NeuralCore
def get_initial_pruning_masks_from_model(model, ir_graph: IRGraph):
    """Collect per-node and per-bank ``(row_mask, col_mask)`` pruning masks from the model layers.

    Keyed by IR node / bank id, aligned to each core's IR ``(axons, neurons)`` convention;
    ``True`` means pruned, bias rows appended where present.
    """
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
            ir_row_mask.append(False)
        ir_row_mask = (ir_row_mask + [False] * nr)[:nr]
        ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
        ir_col_mask = (ir_col_mask + [False] * nc)[:nc]
        if len(ir_row_mask) == nr and len(ir_col_mask) == nc:
            initial_pruned_per_node[node.id] = (ir_row_mask, ir_col_mask)
        else:
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

