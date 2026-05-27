"""Shared helpers for adapting torchvision image-model builders."""

from __future__ import annotations

import torch
import torch.nn as nn


def parse_image_input_shape(input_shape, *, model_name: str) -> tuple[int, int, int]:
    """Validate and normalize a ``(C, H, W)`` input shape."""
    shape = tuple(int(x) for x in input_shape)
    if len(shape) != 3:
        raise ValueError(f"{model_name} expects input_shape (C, H, W), got {shape!r}")
    c, h, w = shape
    if c <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"{model_name} needs positive input dimensions, got {shape!r}")
    return c, h, w


def resize_conv_input_weights(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Project pretrained conv weights to a new input-channel count."""
    if weight.shape[1] == in_channels:
        return weight.detach().clone()
    if in_channels == 1:
        return weight.mean(dim=1, keepdim=True)
    if in_channels < weight.shape[1]:
        return weight[:, :in_channels].detach().clone()
    extra = in_channels - weight.shape[1]
    mean = weight.mean(dim=1, keepdim=True).repeat(1, extra, 1, 1)
    return torch.cat([weight.detach().clone(), mean], dim=1)


def adapt_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Clone ``conv`` with a new input-channel count and projected weights."""
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.copy_(resize_conv_input_weights(conv.weight, in_channels))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv
