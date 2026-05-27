"""Fuse perceptron normalization into the linear layer (training-time)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


def fuse_into_perceptron(perceptron, *, device: torch.device | str) -> None:
    """Fold ``perceptron.normalization`` into ``perceptron.layer``; set norm to Identity."""
    if isinstance(perceptron.normalization, nn.Identity):
        return

    perceptron.to(device)
    pt = PerceptronTransformer()
    u, beta, mean = pt._get_u_beta_mean(perceptron.normalization)

    W = perceptron.layer.weight.data
    b = (
        perceptron.layer.bias.data
        if perceptron.layer.bias is not None
        else torch.zeros(W.shape[0], device=W.device)
    )

    fused_W = W * u.unsqueeze(-1)
    fused_b = (b - mean) * u + beta

    saved_buffers = {
        buf_name: buf_val.clone()
        for buf_name, buf_val in perceptron.layer.named_buffers()
    }

    perceptron.layer = nn.Linear(
        perceptron.input_features,
        perceptron.output_channels,
        bias=True,
    )
    perceptron.layer.weight.data = fused_W
    perceptron.layer.bias.data = fused_b
    for buf_name, buf_val in saved_buffers.items():
        perceptron.layer.register_buffer(buf_name, buf_val)

    perceptron.normalization = nn.Identity()
