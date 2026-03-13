"""Native PyTorch MLP-Mixer (plain nn.Module) for torch_mapping conversion."""

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


class TorchMLPMixer(nn.Module):
    """MLP-Mixer as plain nn.Module: patch embed, 2x (token mixer + channel mixer), pool, classifier.

    Config aligned with the old PerceptronMixerBuilder: patch_n_1, patch_m_1 (patch grid),
    patch_c_1 (patch channels), fc_w_1 (token mixer hidden), fc_w_2 (channel mixer hidden).
    Uses only Linear, Conv2d, BatchNorm1d, ReLU/LeakyReLU/GELU, reshape/permute for torch_mapping.
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
        act = _get_activation(base_activation)

        # Patch embedding: Conv2d then flatten to (B, num_patches, patch_c_1)
        self.patch_embed = nn.Conv2d(
            c, patch_c_1, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w)
        )
        self.patch_bn = nn.BatchNorm2d(patch_c_1)

        # Two mixer blocks (token mixer + channel mixer each)
        self.token_mix_1 = _TokenMixer(num_patches, patch_c_1, fc_w_1, act)
        self.channel_mix_1 = _ChannelMixer(patch_c_1, fc_w_2, act)
        self.token_mix_2 = _TokenMixer(num_patches, patch_c_1, fc_w_1, act)
        self.channel_mix_2 = _ChannelMixer(patch_c_1, fc_w_2, act)

        self.norm = nn.LazyBatchNorm1d()
        self.classifier = nn.Linear(patch_c_1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, patch_c_1, num_patches) after conv+flatten
        x = self.patch_embed(x)
        x = self.patch_bn(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, num_patches, patch_c_1)
        x = self.token_mix_1(x)
        x = self.channel_mix_1(x)
        x = self.token_mix_2(x)
        x = self.channel_mix_2(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, patch_c_1)
        x = self.classifier(x)
        return x


class _TokenMixer(nn.Module):
    """Mix over the patch dimension: (B, N, D) -> (B, N, D) via two Linear(N, fc_w) and back."""

    def __init__(self, num_patches: int, dim: int, hidden: int, act: nn.Module):
        super().__init__()
        self.ln = nn.LazyBatchNorm1d()
        self.fc1 = nn.Linear(num_patches, hidden)
        self.fc2 = nn.Linear(hidden, num_patches)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, N, D)
        x = self.ln(x)
        x = x.permute(0, 2, 1)  # (B, D, N)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 2, 1)  # (B, N, D)
        return x


class _ChannelMixer(nn.Module):
    """Mix over the channel dimension: (B, N, D) -> (B, N, D) via Linear(D, hidden) and back."""

    def __init__(self, dim: int, hidden: int, act: nn.Module):
        super().__init__()
        self.ln = nn.LazyBatchNorm1d()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
