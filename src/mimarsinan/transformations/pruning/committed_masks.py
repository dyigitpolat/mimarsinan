"""Committed-raw-parameter pruning: masks must hold in stored params, not only in call-time hooks (the deployed segment executor never fires module hooks)."""

from __future__ import annotations

import torch


def commit_layer_pruning(layer) -> None:
    """Zero the layer's raw weight/bias entries under its committed prune masks
    (``prune_mask`` / ``prune_bias_mask`` buffers); no-op without masks."""
    prune_mask = getattr(layer, "prune_mask", None)
    if prune_mask is not None:
        with torch.no_grad():
            layer.weight.data[prune_mask] = 0.0
    prune_bias_mask = getattr(layer, "prune_bias_mask", None)
    if prune_bias_mask is not None and getattr(layer, "bias", None) is not None:
        with torch.no_grad():
            layer.bias.data[prune_bias_mask] = 0.0


def commit_norm_pruning(norm) -> None:
    """Zero the norm's ``running_mean``/affine ``bias`` under its committed
    ``_prune_row_mask``; no-op without a mask."""
    mask = getattr(norm, "_prune_row_mask", None)
    if mask is None:
        return
    with torch.no_grad():
        if getattr(norm, "running_mean", None) is not None:
            norm.running_mean.data[mask] = 0.0
        beta = getattr(norm, "bias", None)
        if beta is not None and isinstance(beta, torch.nn.Parameter):
            beta.data[mask] = 0.0


def commit_perceptron_pruning(perceptron) -> None:
    """Commit the perceptron's prune masks into its raw layer + norm parameters."""
    commit_layer_pruning(perceptron.layer)
    norm = getattr(perceptron, "normalization", None)
    if norm is not None:
        commit_norm_pruning(norm)


def commit_model_pruning(model) -> None:
    """Commit every prune mask in ``model``'s module tree into raw parameters —
    the module-tree form of the enforcement hooks, for seams (the pipeline
    cache) that see arbitrary modules rather than perceptron lists."""
    for module in model.modules():
        commit_layer_pruning(module)
        commit_norm_pruning(module)


def _check_zero(tensor, mask, what: str, label: str, where: str) -> None:
    masked = tensor.detach()[mask]
    if masked.numel() == 0:
        return
    max_abs = float(masked.abs().max())
    if max_abs != 0.0:
        raise RuntimeError(
            f"[{where}] pruning contract violated: {label} carries "
            f"non-zero {what} in pruned entries (max |value| = {max_abs:.6g}). "
            "Raw parameters must satisfy the committed prune masks — the "
            "deployed executor never fires the enforcement hooks."
        )


def _verify_layer(layer, label: str, where: str) -> None:
    prune_mask = getattr(layer, "prune_mask", None)
    if prune_mask is not None:
        _check_zero(layer.weight, prune_mask, "weight", label, where)
    prune_bias_mask = getattr(layer, "prune_bias_mask", None)
    if prune_bias_mask is not None and getattr(layer, "bias", None) is not None:
        _check_zero(layer.bias, prune_bias_mask, "bias", label, where)


def _verify_norm(norm, label: str, where: str) -> None:
    row_mask = getattr(norm, "_prune_row_mask", None)
    if row_mask is None:
        return
    running_mean = getattr(norm, "running_mean", None)
    if running_mean is not None:
        _check_zero(running_mean, row_mask, "norm running_mean", label, where)
    beta = getattr(norm, "bias", None)
    if beta is not None and isinstance(beta, torch.nn.Parameter):
        _check_zero(beta, row_mask, "norm bias", label, where)


def verify_committed_pruning(perceptrons, *, where: str) -> None:
    """Fail loud unless ``mask * param == param`` for every committed prune mask
    (weights, biases, and the norm's mean/beta) across ``perceptrons``."""
    for index, perceptron in enumerate(perceptrons):
        label = f"perceptron {index}"
        _verify_layer(perceptron.layer, label, where)
        norm = getattr(perceptron, "normalization", None)
        if norm is not None:
            _verify_norm(norm, label, where)


def verify_model_pruning(model, *, where: str) -> None:
    """Fail loud unless ``mask * param == param`` holds for every committed
    prune mask across ``model``'s module tree (the cache-boundary verify,
    mirroring the mapping-time one)."""
    for name, module in model.named_modules():
        label = name or type(module).__name__
        _verify_layer(module, label, where)
        _verify_norm(module, label, where)
