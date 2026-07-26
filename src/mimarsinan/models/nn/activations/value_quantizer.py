"""[mvm AQ] Symmetric signed value-grid quantization for host↔chip boundaries."""

from __future__ import annotations

import torch
import torch.nn as nn


def value_grid_levels(activation_bits: int) -> int:
    """Positive levels of the symmetric signed grid: 2^(b-1) - 1."""
    return (1 << (int(activation_bits) - 1)) - 1


def quantize_to_value_grid(
    x: torch.Tensor, scale: torch.Tensor, activation_bits: int
) -> torch.Tensor:
    """Round ``x`` onto the signed grid ``±scale`` with ``2^(b-1)-1`` positive
    levels (saturating). ``scale <= 0`` is the unarmed identity. One formula
    for the model-side module and the value executor — the boundary SSOT."""
    levels = value_grid_levels(activation_bits)
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    if levels <= 0 or float(scale.max()) <= 0.0:
        return x
    step = scale / levels
    return torch.clamp(torch.round(x / step), -levels, levels) * step


class ValueGridQuantizer(nn.Module):
    """STE boundary quantizer for value-domain (mvm) segment entries.

    Holds a LIVE reference to the perceptron's ``input_activation_scale``
    (the one-writer currency), so the model forward, WQ QAT, and the IR
    emission all read the same calibrated scale.
    """

    def __init__(self, scale: torch.Tensor, activation_bits: int) -> None:
        super().__init__()
        self.scale = scale
        self.activation_bits = int(activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized = quantize_to_value_grid(x, self.scale, self.activation_bits)
        return x + (quantized - x).detach()
