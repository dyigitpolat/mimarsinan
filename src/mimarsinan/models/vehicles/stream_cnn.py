"""StreamCNN: spiking-native stride-2 conv stack — streamable by construction."""

from __future__ import annotations

import torch.nn as nn


def _activation(name: str) -> nn.Module:
    if name == "LeakyReLU":
        return nn.LeakyReLU(inplace=True)
    return nn.ReLU(inplace=True)


class StreamCNN(nn.Module):
    """Stride-2 Conv-BN-ReLU stages (no pooling), one activated FC block, and
    a bare-Linear readout: host compute ops appear ONLY as the encode prefix
    (first conv under ``subsume``) and the readout suffix — the streamed-lif
    structural contract holds for every configuration of this vehicle.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        width: int = 16,
        blocks: int = 3,
        fc_width: int = 128,
        base_activation: str = "ReLU",
    ):
        super().__init__()
        shape = tuple(input_shape)
        if len(shape) != 3:
            raise ValueError(f"StreamCNN expects input_shape (C, H, W), got {shape}")
        channels, height, w = shape

        layers: list[nn.Module] = []
        in_ch = channels
        out_ch = int(width)
        for _ in range(int(blocks)):
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_activation(base_activation))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 128)
            height = (height + 1) // 2
            w = (w + 1) // 2
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        flat = in_ch * height * w
        self.fc = nn.Linear(flat, int(fc_width))
        self.fc_act = _activation(base_activation)
        self.head = nn.Linear(int(fc_width), int(num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc_act(self.fc(x))
        return self.head(x)
