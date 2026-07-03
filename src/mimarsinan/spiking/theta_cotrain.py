"""Promote per-perceptron ``activation_scale`` (theta) to a trainable per-output-channel Parameter for deployed-cascade fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn


def _scale_carrying_nodes(perceptron):
    """Activation modules under ``perceptron`` (excluding the perceptron itself) that
    carry an ``activation_scale`` — e.g. ``TTFSActivation`` and a blend target."""
    for module in perceptron.modules():
        if module is perceptron:
            continue
        if isinstance(getattr(module, "activation_scale", None), torch.Tensor):
            yield module


def promote_activation_scale_per_channel(model, *, skip_encoding: bool = True) -> list:
    """Rebind each non-encoding perceptron's ``activation_scale`` to a per-channel
    ``requires_grad`` Parameter, on the perceptron and every node that references
    it. Idempotent: a second call re-syncs node references. Returns the new params.
    """
    params = []
    for perceptron in model.get_perceptrons():
        if skip_encoding and getattr(perceptron, "is_encoding_layer", False):
            continue
        scale = perceptron.activation_scale.detach()
        out_dim = int(perceptron.layer.weight.shape[0])
        vec = (
            scale * torch.ones(out_dim, dtype=scale.dtype, device=scale.device)
            if scale.dim() == 0
            else scale.clone()
        )
        param = nn.Parameter(vec.contiguous(), requires_grad=True)
        perceptron.activation_scale = param
        for node in _scale_carrying_nodes(perceptron):
            node.activation_scale = param
        params.append(param)
    return params
