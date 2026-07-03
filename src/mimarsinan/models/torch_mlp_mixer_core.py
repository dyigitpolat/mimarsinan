"""MLP-Mixer Core: variant with activation after every FC so all mixer layers package as perceptrons."""

from __future__ import annotations

import torch
import torch.nn as nn


def _get_activation(name: str):
    if name == "ReLU":
        return nn.ReLU(inplace=True)
    if name == "LeakyReLU":
        return nn.LeakyReLU(inplace=True)
    if name == "GELU":
        return nn.GELU()
    return nn.ReLU(inplace=True)


class _TokenMixerCore(nn.Module):
    """Token mixer with activation after both fc1 and fc2 for perceptron packaging."""

    def __init__(self, num_patches: int, dim: int, hidden: int, act: nn.Module):
        super().__init__()
        self.ln = nn.Identity()
        self.fc1 = nn.Linear(num_patches, hidden)
        self.fc2 = nn.Linear(hidden, num_patches)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = x.permute(0, 2, 1)
        return x


class _ChannelMixerCore(nn.Module):
    """Channel mixer with activation after both fc1 and fc2 for perceptron packaging."""

    def __init__(self, dim: int, hidden: int, act: nn.Module):
        super().__init__()
        self.ln = nn.Identity()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class TorchMLPMixerCore(nn.Module):
    """MLP-Mixer with activation after every FC so all mixer layers package as chip perceptrons.

    Same structure as TorchMLPMixer but applies the base activation after both fc1 and fc2 in each
    block, yielding Linear->act everywhere so the converter emits Perceptrons, not host Identities.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        patch_n_1: int,
        patch_m_1: int,
        patch_c_1: int,
        fc_w_1: int,
        fc_w_2: int,
        base_activation: str = "ReLU",
        num_blocks: int = 2,
    ):
        super().__init__()
        c, h, w = int(input_shape[-3]), int(input_shape[-2]), int(input_shape[-1])
        patch_h = h // patch_n_1
        patch_w = w // patch_m_1
        num_patches = patch_n_1 * patch_m_1
        self.num_patches = num_patches
        self.patch_channels = patch_c_1
        self.num_blocks = int(num_blocks)
        self.patch_embed = nn.Conv2d(
            c, patch_c_1, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w)
        )
        self.patch_bn = nn.BatchNorm2d(patch_c_1)

        act = _get_activation(base_activation)
        self.act = act
        self.mixer_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.mixer_blocks.append(_TokenMixerCore(num_patches, patch_c_1, fc_w_1, act))
            self.mixer_blocks.append(_ChannelMixerCore(patch_c_1, fc_w_2, act))

        self.norm = nn.Identity()
        self.classifier = nn.Linear(patch_c_1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.patch_bn(x)
        x = self.act(x)
        x = x.flatten(2).permute(0, 2, 1)
        for block in self.mixer_blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
