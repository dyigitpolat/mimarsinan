"""Pruning utilities: mask computation and weight pruning application.

This module provides functions to:
1. Compute pruning masks based on activation statistics from sample inference.
2. Apply rate-adaptive pruning to weights using those masks (absolute scaling).
3. Compute cross-layer-aware masks that propagate pruning decisions through
   connected perceptrons.
"""

import torch
import math
import copy


def _collect_activation_stats(model, data_loader, device, num_batches=5):
    """Run sample inference and collect per-perceptron activation statistics.

    For each perceptron, collects:
    - input_importance: Mean absolute value per input feature (column significance)
    - output_importance: Mean absolute value per output neuron (row significance)

    Args:
        model: The Supermodel (or any model with get_perceptrons()).
        data_loader: DataLoader providing input batches.
        device: Torch device.
        num_batches: Number of batches to average over.

    Returns:
        List of dicts with 'input_importance' and 'output_importance' tensors,
        one per perceptron in get_perceptrons() order.
    """
    perceptrons = model.get_perceptrons()
    n = len(perceptrons)

    # Accumulators
    input_sums = [None] * n
    output_sums = [None] * n
    count = 0

    # Register hooks on each perceptron's layer (nn.Linear)
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            # inp is a tuple; the first element is the input tensor
            x = inp[0].detach()
            y = out.detach()

            # Flatten all dims except the feature dim for both input and output
            # Input shape can be (B, in_f) or (B, S, in_f) etc.
            x_flat = x.reshape(-1, x.shape[-1])  # (N, in_features)
            y_flat = y.reshape(-1, y.shape[-1])  # (N, out_features)

            in_imp = x_flat.abs().mean(dim=0)   # (in_features,)
            out_imp = y_flat.abs().mean(dim=0)   # (out_features,)

            if input_sums[idx] is None:
                input_sums[idx] = in_imp
                output_sums[idx] = out_imp
            else:
                input_sums[idx] += in_imp
                output_sums[idx] += out_imp
        return hook_fn

    for i, p in enumerate(perceptrons):
        h = p.layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Run inference
    model.eval()
    with torch.no_grad():
        batch_iter = iter(data_loader)
        for _ in range(num_batches):
            try:
                x, y = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader)
                x, y = next(batch_iter)
            x = x.to(device)
            model(x)
            count += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute averages
    stats = []
    for i in range(n):
        stats.append({
            'input_importance': input_sums[i] / count if input_sums[i] is not None else None,
            'output_importance': output_sums[i] / count if output_sums[i] is not None else None,
        })

    return stats


def compute_pruning_masks_from_activations(stats, perceptron, pruning_fraction):
    """Compute boolean masks identifying rows and columns to prune based on
    activation statistics.

    Row pruning: output neurons with smallest mean absolute post-linear activation.
    Column pruning: input features with smallest mean absolute input activation.

    Args:
        stats: Dict with 'input_importance' and 'output_importance' tensors.
        perceptron: A Perceptron module (used for shape validation).
        pruning_fraction: Float in [0, 1].

    Returns:
        (row_mask, col_mask): Boolean tensors. True = keep, False = prune.
    """
    weight = perceptron.layer.weight.data
    out_features, in_features = weight.shape

    # Row significance from output activations
    row_importance = stats['output_importance']
    n_prune_rows = int(math.floor(out_features * pruning_fraction))
    if n_prune_rows == 0 or row_importance is None:
        row_mask = torch.ones(out_features, dtype=torch.bool, device=weight.device)
    elif n_prune_rows >= out_features:
        row_mask = torch.zeros(out_features, dtype=torch.bool, device=weight.device)
    else:
        _, row_indices = row_importance.to(weight.device).sort()
        row_mask = torch.ones(out_features, dtype=torch.bool, device=weight.device)
        row_mask[row_indices[:n_prune_rows]] = False

    # Column significance from input activations
    col_importance = stats['input_importance']
    n_prune_cols = int(math.floor(in_features * pruning_fraction))
    if n_prune_cols == 0 or col_importance is None:
        col_mask = torch.ones(in_features, dtype=torch.bool, device=weight.device)
    elif n_prune_cols >= in_features:
        col_mask = torch.zeros(in_features, dtype=torch.bool, device=weight.device)
    else:
        _, col_indices = col_importance.to(weight.device).sort()
        col_mask = torch.ones(in_features, dtype=torch.bool, device=weight.device)
        col_mask[col_indices[:n_prune_cols]] = False

    return row_mask, col_mask


def compute_pruning_masks(perceptron, pruning_fraction):
    """Compute boolean masks using weight L1-norm (fallback / legacy).

    Rows correspond to output neurons (dim 0 of weight matrix).
    Columns correspond to input features / axons (dim 1 of weight matrix).

    Args:
        perceptron: A Perceptron module with a ``layer`` attribute (nn.Linear).
        pruning_fraction: Float in [0, 1]. Fraction of rows/columns to prune
            (the least significant ones by L1-norm).

    Returns:
        (row_mask, col_mask): Boolean tensors. True = keep, False = prune.
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


def compute_all_pruning_masks(perceptrons, pruning_fraction, activation_stats=None):
    """Compute pruning masks for all perceptrons with cross-layer propagation.

    When activation_stats is provided, uses activation-based significance.
    Otherwise falls back to weight-L1 significance.

    First computes independent masks per perceptron (guaranteeing at least
    pruning_fraction of rows and columns are pruned). Then propagates:
    if layer L prunes row i (output neuron i is dead), and layer L+1 has
    in_features == out_features of L, then column i of L+1 is also pruned
    (on top of L+1's own column-level pruning).

    Args:
        perceptrons: List of Perceptron modules in forward-topological order.
        pruning_fraction: Float in [0, 1]. Minimum fraction to prune.
        activation_stats: Optional list of dicts from _collect_activation_stats.

    Returns:
        List of (row_mask, col_mask) tuples, one per perceptron.
    """
    # Phase 1: compute independent masks
    masks = []
    for i, p in enumerate(perceptrons):
        if activation_stats is not None and i < len(activation_stats):
            row_mask, col_mask = compute_pruning_masks_from_activations(
                activation_stats[i], p, pruning_fraction
            )
        else:
            row_mask, col_mask = compute_pruning_masks(p, pruning_fraction)
        masks.append((row_mask, col_mask))

    # Exempt input-buffer (first layer columns) and output-buffer (last layer rows)
    n = len(perceptrons)
    if n > 0:
        first_row, first_col = masks[0]
        masks[0] = (first_row, torch.ones_like(first_col, dtype=torch.bool, device=first_col.device))
        last_row, last_col = masks[-1]
        masks[-1] = (torch.ones_like(last_row, dtype=torch.bool, device=last_row.device), last_col)

    # Phase 2: propagate row pruning as additional column pruning
    for i in range(len(perceptrons) - 1):
        prev_row_mask = masks[i][0]  # output neurons of layer i
        next_col_mask = masks[i + 1][1]  # input features of layer i+1

        prev_out = prev_row_mask.shape[0]
        next_in = next_col_mask.shape[0]

        # Only propagate when dimensions match (direct connection)
        if prev_out == next_in:
            # Where prev layer pruned a row, also prune the corresponding column
            # in the next layer (union with existing column pruning)
            propagated_prune = ~prev_row_mask  # True where prev row was pruned
            combined_col_mask = next_col_mask & ~propagated_prune
            masks[i + 1] = (masks[i + 1][0], combined_col_mask)

    return masks


def apply_pruning_masks(perceptron, row_mask, col_mask, rate, original_weight, original_bias):
    """Apply rate-adaptive pruning using absolute scaling from original weights.

    For each weight at position [i, j]:
        - If row i is pruned OR column j is pruned:
            weight[i,j] = original_weight[i,j] * (1.0 - rate)

    At rate=0.0, weights equal originals. At rate=1.0, pruned weights are zeroed.
    Unpruned weights are always restored to their original+trained values.

    Args:
        perceptron: A Perceptron module.
        row_mask: Boolean tensor (out_features,). True = keep, False = prune.
        col_mask: Boolean tensor (in_features,). True = keep, False = prune.
        rate: Float in [0, 1]. Adaptation rate.
        original_weight: The original weight tensor before pruning started.
        original_bias: The original bias tensor before pruning started (or None).
    """
    if rate == 0.0:
        return

    weight = perceptron.layer.weight.data  # (out_features, in_features)

    # Build a 2D mask: True where the weight is in a pruned row or column
    pruned_rows = ~row_mask  # (out_features,)
    pruned_cols = ~col_mask  # (in_features,)
    # Outer OR: a weight is affected if its row OR its column is pruned
    prune_mask = pruned_rows.unsqueeze(1) | pruned_cols.unsqueeze(0)  # (out, in)

    # Absolute scaling: pruned positions move toward zero from their original
    # values while kept positions are untouched
    scale = 1.0 - rate
    weight[prune_mask] = original_weight[prune_mask] * scale

    # Also scale bias for pruned rows
    if perceptron.layer.bias is not None and original_bias is not None:
        perceptron.layer.bias.data[pruned_rows] = original_bias[pruned_rows] * scale
