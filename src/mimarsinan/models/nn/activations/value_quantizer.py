"""[mvm AQ] Symmetric signed value-grid quantization for host↔chip boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def value_grid_levels(activation_bits: int) -> int:
    """Positive levels of the symmetric signed grid: 2^(b-1) - 1."""
    return (1 << (int(activation_bits) - 1)) - 1


@dataclass(frozen=True)
class BoundaryGrid:
    """The realized value grid at ONE host→chip boundary.

    A first-class descriptor, deliberately distinct from the event domain's
    ``input_activation_scale`` wire currency: overloading that float left the
    two meanings indistinguishable by type and forced a domain conditional at
    every writer.
    """

    scale: float
    bits: int

    @property
    def levels(self) -> int:
        return value_grid_levels(self.bits)

    @property
    def step(self) -> float:
        """One LSB — the unit the AQ certificate is judged in."""
        levels = self.levels
        return (self.scale / levels) if levels > 0 else 0.0

    @property
    def armed(self) -> bool:
        return self.scale > 0.0 and self.levels > 0


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

    Owns its grid scale as a DECLARED buffer: it travels with the module
    through ``.to()`` / deepcopy / state_dict without aliasing another
    module's parameter, and the IR emission reads the grid from here rather
    than from a shared activation-scale slot.
    """

    scale: torch.Tensor  # declared so the buffer types as a Tensor

    def __init__(self, activation_bits: int, scale: float = 0.0) -> None:
        super().__init__()
        self.activation_bits = int(activation_bits)
        self.register_buffer("scale", torch.tensor(float(scale)))

    def calibrate(self, scale: float) -> None:
        """Install the measured boundary range (the single write path)."""
        self.scale.fill_(float(scale))

    @property
    def grid(self) -> BoundaryGrid:
        return BoundaryGrid(scale=float(self.scale), bits=self.activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized = quantize_to_value_grid(x, self.scale, self.activation_bits)
        return x + (quantized - x).detach()


def boundary_grid_of(module: "nn.Module | None") -> "BoundaryGrid | None":
    """The armed grid an entry's input-wire chain realizes, if any."""
    if module is None:
        return None
    for candidate in module.modules():
        if isinstance(candidate, ValueGridQuantizer) and candidate.grid.armed:
            return candidate.grid
    return None
