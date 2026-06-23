"""LeNet5: the classic LeNet-5 CNN, adapted to the dataset input shape."""

from __future__ import annotations

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU(inplace=True)
    if name == "LeakyReLU":
        return nn.LeakyReLU(inplace=True)
    if name == "GELU":
        return nn.GELU()
    return nn.ReLU(inplace=True)


def _conv_output_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
    """Output size after Conv2d/MaxPool2d: (size + 2*padding - kernel_size) // stride + 1."""
    return (size + 2 * padding - kernel_size) // stride + 1


class LeNet5(nn.Module):
    """Classic LeNet-5: Conv(1->6,k5)->act->pool->Conv(6->16,k5)->act->pool->FC120->act->FC84->act->FC.

    Pipeline-native: uses only on-chip-or-structural ops (Conv2d, ReLU, MaxPool2d, Linear).
    No grouped/depthwise convolutions. Input channels and the flattened FC size are
    derived from ``input_shape`` so the model adapts to e.g. MNIST 1x28x28.

    The k5 convs use ``padding=2`` (SAME): this is the canonical "pad the 28x28 MNIST
    image so the k5 stages fit" framing, and it keeps the second multi-channel conv on
    the mappable (padded) path — valid (no-pad) multi-channel convs hit a LayoutSourceView
    symbolic-index limitation in the soft-core mapper.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        base_activation: str = "ReLU",
    ):
        super().__init__()
        shape = tuple(input_shape)
        if len(shape) != 3:
            raise ValueError(f"LeNet5 expects input_shape (C, H, W), got {shape}")
        in_channels, h_in, w_in = int(shape[0]), int(shape[1]), int(shape[2])

        kernel_size = 5
        padding = 2
        pool_kernel = 2

        h = _conv_output_size(h_in, kernel_size, 1, padding)
        w = _conv_output_size(w_in, kernel_size, 1, padding)
        h = _conv_output_size(h, pool_kernel, pool_kernel, 0)
        w = _conv_output_size(w, pool_kernel, pool_kernel, 0)
        h = _conv_output_size(h, kernel_size, 1, padding)
        w = _conv_output_size(w, kernel_size, 1, padding)
        h = _conv_output_size(h, pool_kernel, pool_kernel, 0)
        w = _conv_output_size(w, pool_kernel, pool_kernel, 0)
        if h <= 0 or w <= 0:
            raise ValueError(
                f"LeNet5: conv+pool output spatial size would be ({h}, {w}); "
                f"input {shape} is too small for two k{kernel_size} conv + pool stages."
            )
        flat_size = 16 * h * w

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=kernel_size, padding=padding),
            _get_activation(base_activation),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel),
            nn.Conv2d(6, 16, kernel_size=kernel_size, padding=padding),
            _get_activation(base_activation),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 120),
            _get_activation(base_activation),
            nn.Linear(120, 84),
            _get_activation(base_activation),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
