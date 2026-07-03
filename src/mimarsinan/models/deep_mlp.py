"""DeepMLP: narrow, configurable-depth Linear+ReLU stack (the depth-probe vehicle)."""

from __future__ import annotations

import torch
import torch.nn as nn


_RESIDUAL_BLOCK_SIZE = 2


def _get_activation(name: str) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU(inplace=True)
    if name == "LeakyReLU":
        return nn.LeakyReLU(inplace=True)
    if name == "GELU":
        return nn.GELU()
    return nn.ReLU(inplace=True)


class DeepMLP(nn.Module):
    """Flatten -> [Linear(width) + activation] x depth -> Linear(num_classes); the depth-probe vehicle.

    Converts through the same torch->perceptron path as the MLP-Mixer. ``residual`` (opt-in)
    adds equal-width additive skips that share a state_dict with the plain model, because a
    bare equal-width add lowers to a param-free host ComputeOp and leaves the Linear names unchanged.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        depth: int,
        width: int = 64,
        base_activation: str = "ReLU",
        residual: bool = False,
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
        self.residual = bool(residual)
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

    def _hidden_layer(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply the ``idx``-th hidden Linear+activation pair (``hidden[2*idx:2*idx+2]``)."""
        return self.hidden[2 * idx + 1](self.hidden[2 * idx](x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        if not self.residual:
            return self.classifier(self.hidden(x))

        x = self._hidden_layer(x, 0)
        idx = 1
        while idx + _RESIDUAL_BLOCK_SIZE <= self.depth:
            block = x
            for offset in range(_RESIDUAL_BLOCK_SIZE):
                block = self._hidden_layer(block, idx + offset)
            x = x + block
            idx += _RESIDUAL_BLOCK_SIZE
        while idx < self.depth:
            x = self._hidden_layer(x, idx)
            idx += 1
        return self.classifier(x)
