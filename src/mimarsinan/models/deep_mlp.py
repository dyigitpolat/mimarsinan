"""DeepMLP: narrow, configurable-depth Linear+ReLU stack (the depth-probe vehicle)."""

from __future__ import annotations

import torch
import torch.nn as nn


_RESIDUAL_BLOCK_SIZE = 2  # consecutive hidden layers wrapped by one equal-width skip


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

    ``residual`` (opt-in, default off) wraps consecutive pairs of equal-width
    hidden layers in an additive skip ``z = z + block(z)``. The FIRST hidden layer
    maps ``input_size -> width`` (a stem/projection), so it is applied plainly; only
    the equal-width hidden layers that follow are residual-wrapped, making each skip
    a bare equal-width add. Bare equal-width adds lower to a param-free host
    ``ComputeAdapter(operator.add)`` ComputeOp (the residual-mapping path) and leave
    the Linear count and ``hidden.*`` parameter names unchanged, so a plain and a
    residual model share a state_dict. An odd trailing equal-width hidden layer is
    applied without a skip.
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

        x = self._hidden_layer(x, 0)  # stem: input_size -> width (not equal-width; no skip)
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
