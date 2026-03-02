"""Pruning utilities: mask computation and weight pruning application.

This module provides functions to:
1. Compute pruning masks for perceptron weights based on L1-norm significance.
2. Apply rate-adaptive pruning to weights using those masks.
"""

import torch
import math


def compute_pruning_masks(perceptron, pruning_fraction):
    """Compute boolean masks identifying rows and columns to prune.

    Rows correspond to output neurons (dim 0 of weight matrix).
    Columns correspond to input features / axons (dim 1 of weight matrix).

    Args:
        perceptron: A Perceptron module with a ``layer`` attribute (nn.Linear).
        pruning_fraction: Float in [0, 1]. Fraction of rows/columns to prune
            (the least significant ones by L1-norm).

    Returns:
        (row_mask, col_mask): Boolean tensors. True = keep, False = prune.
            row_mask has shape (out_features,), col_mask has shape (in_features,).
    """
    weight = perceptron.layer.weight.data  # (out_features, in_features)
    out_features, in_features = weight.shape

    # Row significance: L1-norm across columns for each row
    row_l1 = weight.abs().sum(dim=1)  # (out_features,)
    n_prune_rows = int(math.floor(out_features * pruning_fraction))
    if n_prune_rows == 0:
        row_mask = torch.ones(out_features, dtype=torch.bool, device=weight.device)
    elif n_prune_rows >= out_features:
        row_mask = torch.zeros(out_features, dtype=torch.bool, device=weight.device)
    else:
        _, row_indices = row_l1.sort()
        row_mask = torch.ones(out_features, dtype=torch.bool, device=weight.device)
        row_mask[row_indices[:n_prune_rows]] = False

    # Column significance: L1-norm across rows for each column
    col_l1 = weight.abs().sum(dim=0)  # (in_features,)
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


def apply_pruning_masks(perceptron, row_mask, col_mask, rate):
    """Apply rate-adaptive pruning to a perceptron's weights.

    For each weight at position [i, j]:
        - If row i is pruned OR column j is pruned, multiply by (1.0 - rate).

    At rate=0.0, no change. At rate=1.0, all pruned weights are zeroed.

    Args:
        perceptron: A Perceptron module.
        row_mask: Boolean tensor (out_features,). True = keep, False = prune.
        col_mask: Boolean tensor (in_features,). True = keep, False = prune.
        rate: Float in [0, 1]. Adaptation rate.
    """
    if rate == 0.0:
        return

    weight = perceptron.layer.weight.data  # (out_features, in_features)

    # Build a 2D mask: True where the weight is in a pruned row or column
    pruned_rows = ~row_mask  # (out_features,)
    pruned_cols = ~col_mask  # (in_features,)
    # Outer OR: a weight is affected if its row OR its column is pruned
    prune_mask = pruned_rows.unsqueeze(1) | pruned_cols.unsqueeze(0)  # (out, in)

    scale = 1.0 - rate
    weight[prune_mask] *= scale

    # Also scale bias for pruned rows if bias exists
    if perceptron.layer.bias is not None:
        perceptron.layer.bias.data[pruned_rows] *= scale
