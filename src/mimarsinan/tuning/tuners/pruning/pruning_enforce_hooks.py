"""Module-level forward hooks for persistent pruning enforcement."""

import torch
import torch.nn as nn


def pruning_enforce_linear_pre_hook(module, inputs):
    prune_mask = getattr(module, "prune_mask", None)
    if prune_mask is not None:
        module.weight.data[prune_mask] = 0.0
    prune_bias_mask = getattr(module, "prune_bias_mask", None)
    if prune_bias_mask is not None and module.bias is not None:
        module.bias.data[prune_bias_mask] = 0.0


def pruning_enforce_norm_pre_hook(module, inputs):
    mask = getattr(module, "_prune_row_mask", None)
    if mask is None:
        return
    if getattr(module, "running_mean", None) is not None:
        module.running_mean.data[mask] = 0.0
    beta = getattr(module, "bias", None)
    if beta is not None and isinstance(beta, torch.nn.Parameter):
        beta.data[mask] = 0.0
