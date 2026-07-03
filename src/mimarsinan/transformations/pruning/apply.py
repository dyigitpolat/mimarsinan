"""Apply pruning masks to perceptron weights."""


def apply_pruning_masks(perceptron, row_mask, col_mask, rate, original_weight, original_bias):
    """Apply rate-adaptive pruning using absolute scaling from original weights."""
    if rate == 0.0:
        return

    weight = perceptron.layer.weight.data

    pruned_rows = ~row_mask
    pruned_cols = ~col_mask
    prune_mask = pruned_rows.unsqueeze(1) | pruned_cols.unsqueeze(0)

    scale = 1.0 - rate
    weight[prune_mask] = original_weight[prune_mask] * scale

    if perceptron.layer.bias is not None and original_bias is not None:
        perceptron.layer.bias.data[pruned_rows] = original_bias[pruned_rows] * scale
