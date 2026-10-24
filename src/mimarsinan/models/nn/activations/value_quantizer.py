"""[mvm AQ] Symmetric signed value-grid quantization for host↔chip boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.models.nn.activations.integer_law import grid_levels, round_half_away

#: Entry rounding of a declared boundary grid: the integer law's half-away rule, or the floor of an input encoder.
ENTRY_ROUNDINGS = ("nearest", "floor")


class BoundaryGridDeclarationError(ToolchainRefusal, ValueError):
    """A declared boundary grid the entry's DAC cannot hold (codes, rounding, offset or bits)."""


def value_grid_levels(activation_bits: int) -> int:
    """Positive levels of the symmetric signed grid: 2^(b-1) - 1."""
    return grid_levels(activation_bits)


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

    def entry_offset(self) -> float:
        """The value of code 0 is ``-entry_offset()``: a value is ``code * step - entry_offset()``."""
        return 0.0

    def snap(self, x: torch.Tensor) -> torch.Tensor:
        """The entry snap of ``x`` onto this grid (the boundary SSOT)."""
        return quantize_to_value_grid(x, self.scale, self.bits)

    def entry_codes(self, values: np.ndarray) -> np.ndarray:
        """The DAC codes (int64) the entry snap gives the host ``values``."""
        quotient = torch.as_tensor(values, dtype=torch.float64) / float(self.step)
        return torch.clamp(round_half_away(quotient), -self.levels, self.levels).numpy().astype(np.int64)


@dataclass(frozen=True)
class DeclaredBoundaryGrid(BoundaryGrid):
    """A boundary grid declared with its offset, entry rounding and code range (WI-5a): the snap writes
    ``clamp(rnd((x + offset) / step), code_min, code_max)``, the value ``code * step - offset``. Two grids that
    differ in any field never share a core; ``state_sha256`` names the declared map it came from."""

    offset: float = 0.0
    rounding: str = "nearest"
    code_min: Optional[int] = None
    code_max: Optional[int] = None
    state_sha256: str = ""
    #: A lane chip's DAC (WI-5e) also reads the 2^b unsigned codes ``0..2^b - 1`` (one more than the symmetric grid).
    unsigned_dac: bool = False

    def __post_init__(self) -> None:
        if self.rounding not in ENTRY_ROUNDINGS:
            raise BoundaryGridDeclarationError(
                f"boundary grid rounding {self.rounding!r} is not one of {list(ENTRY_ROUNDINGS)}")
        if not math.isfinite(float(self.offset)):
            raise BoundaryGridDeclarationError(f"boundary grid offset {self.offset!r} is not a finite number")
        low, high = self.code_range
        unsigned = self.unsigned
        if low < -self.levels and not unsigned:
            raise BoundaryGridDeclarationError(
                f"boundary grid code_min {low} is below the {self.bits}-bit DAC's -{self.levels}")
        if high > self.levels and not unsigned:
            raise BoundaryGridDeclarationError(
                f"boundary grid code_max {high} is above the {self.bits}-bit DAC's {self.levels}"
                + ("" if self.unsigned_dac else " (a lane chip's DAC also reads unsigned codes up to "
                                                 f"{(1 << int(self.bits)) - 1}: onchip_arithmetic)"))
        if low >= high:
            raise BoundaryGridDeclarationError(f"boundary grid code_min {low} is not below code_max {high}")

    def entry_offset(self) -> float:
        return float(self.offset)

    @property
    def unsigned(self) -> bool:
        """An unsigned code range on a lane chip's DAC: the unsigned lane grid of ``bits`` (codes within
        ``0..2^b - 1``), whose step spans the whole range, ``scale / (2^b - 1)``, as a lane's readout range does."""
        low, high = self.code_range
        return self.unsigned_dac and low >= 0 and high <= (1 << int(self.bits)) - 1

    @property
    def step(self) -> float:
        if self.unsigned:
            return self.scale / ((1 << int(self.bits)) - 1)
        return super().step

    @property
    def code_range(self) -> Tuple[int, int]:
        low = -self.levels if self.code_min is None else int(self.code_min)
        high = self.levels if self.code_max is None else int(self.code_max)
        return low, high

    def _codes(self, x: torch.Tensor, step: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        quotient = (x + offset) / step
        rounded = torch.floor(quotient) if self.rounding == "floor" else round_half_away(quotient)
        low, high = self.code_range
        return torch.clamp(rounded, low, high)

    def snap(self, x: torch.Tensor) -> torch.Tensor:
        step = torch.as_tensor(self.step, dtype=x.dtype, device=x.device)
        offset = torch.as_tensor(self.offset, dtype=x.dtype, device=x.device)
        return self._codes(x, step, offset) * step - offset

    def entry_codes(self, values: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(values, dtype=torch.float64)
        step = torch.as_tensor(self.step, dtype=torch.float64)
        offset = torch.as_tensor(self.offset, dtype=torch.float64)
        return self._codes(x, step, offset).numpy().astype(np.int64)


def quantize_to_value_grid(
    x: torch.Tensor, scale: "torch.Tensor | float", activation_bits: int
) -> torch.Tensor:
    """Round ``x`` onto the signed grid ``±scale`` with ``2^(b-1)-1`` positive
    levels (saturating), ties away from zero (the integer law's rounding).
    ``scale <= 0`` is the unarmed identity. One formula for the model-side
    module and the value executor — the boundary SSOT."""
    levels = value_grid_levels(activation_bits)
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    if levels <= 0 or float(scale.max()) <= 0.0:
        return x
    step = scale / levels
    return torch.clamp(round_half_away(x / step), -levels, levels) * step


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

    #: A declared grid (WI-5a), set only by ``declare``; absent, the quantizer's state is the calibrated one.
    _declared: Optional[DeclaredBoundaryGrid] = None

    def calibrate(self, scale: float) -> None:
        """Install the measured boundary range (the single write path of a calibrated grid)."""
        self.__dict__.pop("_declared", None)
        self.scale.fill_(float(scale))

    def declare(self, grid: DeclaredBoundaryGrid) -> None:
        """Install a declared grid in place of the calibrated one."""
        if int(grid.bits) != self.activation_bits:
            raise BoundaryGridDeclarationError(
                f"a declared grid of {grid.bits} bits on a boundary quantizer of {self.activation_bits} bits")
        self.scale.fill_(float(grid.scale))
        self._declared = grid

    @property
    def grid(self) -> BoundaryGrid:
        if self._declared is not None:
            return self._declared
        return BoundaryGrid(scale=float(self.scale), bits=self.activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._declared is not None:
            quantized = self._declared.snap(x)
        else:
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
