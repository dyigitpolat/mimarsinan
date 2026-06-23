"""DeepMLP: narrow, configurable-depth Linear+ReLU stack (the depth-probe vehicle)."""

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


class DeepMLP(nn.Module):
    """Flatten -> [Linear(width) + activation] x depth -> Linear(num_classes).

    Pure ``nn.Linear`` + activation (no conv/attention) so it converts through the
    same torch->perceptron path as the MLP-Mixer. ``depth`` is the number of hidden
    layers and directly sets the on-chip cascade depth; this is the T0 vehicle for
    the depth x firing-gain probe.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        depth: int,
        width: int = 64,
        base_activation: str = "ReLU",
    ):
        super().__init__()
        depth = int(depth)
        width = int(width)
        if depth < 1:
            raise ValueError(f"DeepMLP depth must be >= 1, got {depth}.")
        if width < 1:
            raise ValueError(f"DeepMLP width must be >= 1, got {width}.")

        self.depth = depth
        self.width = width
        input_size = int(torch.Size(tuple(input_shape)).numel())

        self.flatten = nn.Flatten()
        layers: list[nn.Module] = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(_get_activation(base_activation))
            in_features = width
        self.hidden = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.hidden(x)
        return self.classifier(x)
