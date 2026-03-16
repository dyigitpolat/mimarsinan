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
        x = x.permute(0, 2, 1)  # (B, D, N)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x)) # Activation for perceptron packaging
        x = x.permute(0, 2, 1)  # (B, N, D)
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
        x = self.act(self.fc2(x)) # Activation for perceptron packaging
        return x


class TorchMLPMixerCore(nn.Module):
    """MLP-Mixer with activation after every FC so all mixer layers can be packaged as chip perceptrons.

    Same structure as TorchMLPMixer but uses _TokenMixerCore and _ChannelMixerCore,
    which apply the base activation after both fc1 and fc2. This yields Linear→act
    for every FC, so the torch_mapping converter creates chip-supported Perceptrons
    instead of host-side Identity layers for the second FC in each block.
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
    ):
        super().__init__()
        c, h, w = int(input_shape[-3]), int(input_shape[-2]), int(input_shape[-1])
        patch_h = h // patch_n_1
        patch_w = w // patch_m_1
        num_patches = patch_n_1 * patch_m_1
        self.num_patches = num_patches
        self.patch_channels = patch_c_1
        self.patch_embed = nn.Conv2d(
            c, patch_c_1, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w)
        )
        self.patch_bn = nn.BatchNorm2d(patch_c_1)

        act = _get_activation(base_activation)
        self.act = act
        self.token_mix_1 = _TokenMixerCore(num_patches, patch_c_1, fc_w_1, act)
        self.channel_mix_1 = _ChannelMixerCore(patch_c_1, fc_w_2, act)
        self.token_mix_2 = _TokenMixerCore(num_patches, patch_c_1, fc_w_1, act)
        self.channel_mix_2 = _ChannelMixerCore(patch_c_1, fc_w_2, act)

        self.norm = nn.Identity()
        self.classifier = nn.Linear(patch_c_1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.patch_bn(x)
        x = self.act(x) # Activation for perceptron packaging
        x = x.flatten(2).permute(0, 2, 1)  # (B, num_patches, patch_c_1)
        x = self.token_mix_1(x)
        x = self.channel_mix_1(x)
        x = self.token_mix_2(x)
        x = self.channel_mix_2(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, patch_c_1)
        x = self.classifier(x)
        return x
