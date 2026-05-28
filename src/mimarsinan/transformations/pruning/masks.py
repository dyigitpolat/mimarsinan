"""Pruning mask computation with cross-layer propagation."""

import math
import torch


def compute_pruning_masks(perceptron, pruning_fraction):
    """Compute boolean masks using weight L1-norm (fallback / legacy)."""
    weight = perceptron.layer.weight.data
    out_features, in_features = weight.shape

    row_l1 = weight.abs().sum(dim=1)
    n_prune_rows = int(math.floor(out_features * pruning_fraction))
    if n_prune_rows == 0:
        row_mask = torch.ones(out_features, dtype=torch.bool, device=weight.device)
    elif n_prune_rows >= out_features:
        row_mask = torch.zeros(out_features, dtype=torch.bool, device=weight.device)
    else:
        _, row_indices = row_l1.sort()
        row_mask = torch.ones(out_features, dtype=torch.bool, device=weight.device)
        row_mask[row_indices[:n_prune_rows]] = False

    col_l1 = weight.abs().sum(dim=0)
    n_prune_cols = int(math.floor(in_features * pruning_fraction))
    if n_prune_cols == 0:
        col_mask = torch.ones(in_features, dtype=torch.bool, device=weight.device)
    elif n_prune_cols >= in_features:
        col_mask = torch.zeros(in_features, dtype=torch.bool, device=weight.device)
    else:
        _, col_indices = col_l1.sort()
        col_mask = torch.ones(in_features, dtype=torch.bool, device=weight.device)
        col_mask[col_indices[:n_prune_cols]] = False

    return row_mask, col_mask


def compute_masks_from_importance(
    perceptrons,
    rate,
    pruning_fraction,
    base_row_imp,
    base_col_imp,
    exempt_input_layers,
    exempt_output_layers,
):
    """Compute per-layer pruning masks from importance scores with cross-layer propagation."""
    import math as _math

    n_layers = len(perceptrons)
    row_masks = []
    col_masks = []
    device = perceptrons[0].layer.weight.device
    for i, p in enumerate(perceptrons):
        out_f, in_f = p.layer.weight.data.shape
        k_r = int(_math.floor(rate * pruning_fraction * out_f))
        if i in exempt_output_layers:
            k_r = 0
        rm = torch.ones(out_f, dtype=torch.bool, device=device)
        if k_r > 0 and i < len(base_row_imp):
            _, idx = base_row_imp[i].to(device).sort()
            rm[idx[:k_r]] = False
        row_masks.append(rm)

        k_c = int(_math.floor(rate * pruning_fraction * in_f))
        if i in exempt_input_layers:
            k_c = 0
        cm = torch.ones(in_f, dtype=torch.bool, device=device)
        if k_c > 0 and i < len(base_col_imp):
            _, idx = base_col_imp[i].to(device).sort()
            cm[idx[:k_c]] = False
        col_masks.append(cm)

    for i in range(n_layers - 1):
        if row_masks[i].shape[0] == col_masks[i + 1].shape[0]:
            col_masks[i + 1] = col_masks[i + 1] & row_masks[i]
    return row_masks, col_masks


def compute_all_pruning_masks(
    perceptrons,
    pruning_fraction,
    exempt_input_layers,
    exempt_output_layers,
    activation_stats=None,
):
    """Compute pruning masks for all perceptrons with cross-layer propagation."""
    base_row_imp = []
    base_col_imp = []
    for i, p in enumerate(perceptrons):
        w = p.layer.weight.data
        if activation_stats is not None and i < len(activation_stats):
            if activation_stats[i].get("output_importance") is not None:
                base_row_imp.append(activation_stats[i]["output_importance"].clone())
            else:
                base_row_imp.append(w.abs().sum(dim=1))
            if activation_stats[i].get("input_importance") is not None:
                base_col_imp.append(activation_stats[i]["input_importance"].clone())
            else:
                base_col_imp.append(w.abs().sum(dim=0))
        else:
            base_row_imp.append(w.abs().sum(dim=1))
            base_col_imp.append(w.abs().sum(dim=0))
    row_masks, col_masks = compute_masks_from_importance(
        perceptrons,
        1.0,
        pruning_fraction,
        base_row_imp,
        base_col_imp,
        exempt_input_layers=exempt_input_layers,
        exempt_output_layers=exempt_output_layers,
    )
    return list(zip(row_masks, col_masks))


def compute_all_pruning_masks_for_ir(
    perceptrons,
    pruning_fraction,
    ir_graph,
    activation_stats=None,
):
    """Compute masks with model-I/O exemption derived from ``ir_graph``."""
    from mimarsinan.mapping.pruning.boundary_policy import (
        compute_perceptron_io_exemption_indices,
    )

    exempt_in, exempt_out = compute_perceptron_io_exemption_indices(
        ir_graph, perceptrons
    )
    return compute_all_pruning_masks(
        perceptrons,
        pruning_fraction,
        exempt_in,
        exempt_out,
        activation_stats=activation_stats,
    )
