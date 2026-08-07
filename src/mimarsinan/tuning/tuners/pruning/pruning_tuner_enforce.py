"""Persistent pruning enforcement after adaptation completes."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.transformations.pruning.seed_generators import (
    install_seed_masks,
    structured_seed_masks_from_keep,
)
from mimarsinan.tuning.tuners.pruning.pruning_enforce_hooks import (
    pruning_enforce_linear_pre_hook,
    pruning_enforce_norm_pre_hook,
)


def register_prune_buffers(perceptrons, row_masks, col_masks):
    """Commit (row_keep, col_keep) masks as layer prune buffers via the shared
    seed-mask SSOT (``transformations.pruning.seed_generators``)."""
    install_seed_masks(
        perceptrons, structured_seed_masks_from_keep(row_masks, col_masks)
    )


def enforce_pruning_persistently(perceptrons, row_masks, col_masks):
    for i, p in enumerate(perceptrons):
        layer = p.layer
        w = layer.weight
        pruned_rows = (~row_masks[i]).to(device=w.device)
        pruned_cols = (~col_masks[i]).to(device=w.device)
        prune_mask_2d = pruned_rows.unsqueeze(1) | pruned_cols.unsqueeze(0)

        with torch.no_grad():
            w.data[prune_mask_2d] = 0.0
            if layer.bias is not None:
                layer.bias.data[pruned_rows] = 0.0

        layer.register_forward_pre_hook(pruning_enforce_linear_pre_hook)
        assert getattr(layer, "prune_mask", None) is not None, (
            "register_prune_buffers must run before enforce_pruning_persistently "
            "so the committed masks back the enforcement hooks"
        )

        norm = getattr(p, "normalization", None)
        if norm is None or isinstance(norm, nn.Identity):
            continue

        norm.register_buffer("_prune_row_mask", pruned_rows.clone())

        if getattr(norm, "running_mean", None) is not None:
            with torch.no_grad():
                norm.running_mean.data[pruned_rows] = 0.0
        beta = getattr(norm, "bias", None)
        if beta is not None and isinstance(beta, torch.nn.Parameter):
            with torch.no_grad():
                beta.data[pruned_rows] = 0.0

        norm.register_forward_pre_hook(pruning_enforce_norm_pre_hook)
