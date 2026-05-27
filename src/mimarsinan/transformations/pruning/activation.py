"""Activation-based pruning mask computation."""

import math
import torch


def collect_activation_stats(model, data_loader, device, num_batches=5):
    """Run sample inference and collect per-perceptron activation statistics."""
    perceptrons = model.get_perceptrons()
    n = len(perceptrons)

    input_sums = [None] * n
    output_sums = [None] * n
    count = 0

    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach()
            y = out.detach()

            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])

            in_imp = x_flat.abs().mean(dim=0)
            out_imp = y_flat.abs().mean(dim=0)

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

    for h in hooks:
        h.remove()

    stats = []
    for i in range(n):
        stats.append({
            'input_importance': input_sums[i] / count if input_sums[i] is not None else None,
            'output_importance': output_sums[i] / count if output_sums[i] is not None else None,
        })

    return stats


def compute_pruning_masks_from_activations(stats, perceptron, pruning_fraction):
    """Compute boolean masks from activation statistics."""
    weight = perceptron.layer.weight.data
    out_features, in_features = weight.shape

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
