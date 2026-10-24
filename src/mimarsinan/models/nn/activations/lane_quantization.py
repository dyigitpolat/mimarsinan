"""Lanes in the model (WI-5e): unsigned and offset lane grids, the fused-add and pool lane ends, and their NF laws."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.nn.activations.integer_law import apply_law, grid_codes, round_half_away


def unsigned_levels(bits: int) -> int:
    """The top code of an unsigned lane grid of ``bits``: ``2^b - 1`` (codes ``0..2^b - 1``, 2^b of them)."""
    return (1 << int(bits)) - 1


def quantize_to_lane_grid(x: torch.Tensor, scale, bits: int, offset: float = 0.0) -> torch.Tensor:
    """Round ``x`` onto the unsigned lane grid ``k Δ - o``, ``Δ = scale / (2^b - 1)``, codes clamped to
    ``[0, 2^b - 1]``, ties away from zero (the law's rounding); ``scale <= 0`` is the unarmed identity."""
    top = unsigned_levels(bits)
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    if float(scale.max()) <= 0.0:
        return x
    step = scale / top
    shift = torch.as_tensor(float(offset), dtype=x.dtype, device=x.device)
    return torch.clamp(round_half_away((x + shift) / step), 0, top) * step - shift


class LaneSeam(nn.Module):
    """The calibration seam of a lane whose rounding sits downstream of its perceptron (a fused add)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LaneAddModule(nn.Module):
    """A fused add lane's end: STE onto the lane grid of its accumulator perceptron's quantizer, or, once
    stamped, that perceptron's law with the other input as the lane's operand (one rounding for the sum)."""

    quantizer: Any  # the accumulator perceptron's ReadoutQuantizer (``end_lane``)

    def __init__(self, add: nn.Module, quantizer: nn.Module, operand_index: int) -> None:
        super().__init__()
        self.add = add
        self.quantizer = quantizer
        self.operand_index = int(operand_index)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.quantizer.end_lane(self.add(*inputs), inputs[self.operand_index])


class LaneReduceModule(nn.Module):
    """A producer-side reduction's end (an average pool its perceptron alone feeds): STE of the pooled value onto the
    perceptron's lane grid, or, once stamped, that perceptron's law on its accumulators summed over each window."""

    quantizer: Any  # the producer perceptron's ReadoutQuantizer (``end_lane``)

    def __init__(self, pool: nn.Module, quantizer: nn.Module) -> None:
        super().__init__()
        self.pool = pool
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer.end_lane(self.pool(x), None)


@dataclass(eq=False)
class PoolStamp:
    """A pool lane's law per channel: codes on the producer's grid, a sum over each window (``divisor`` of
    them, zero padding counted), then ``clamp(rnd_s(M0 Σ q + B - M_o n), lo, hi)``, the value ``y Δout - o``."""

    input_step: float
    input_offset: float
    kernel: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    law: Dict[str, Any]
    _cache: Dict[Any, Any] = field(default_factory=dict, repr=False)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    def _sums(self, codes: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(codes, self.kernel, self.stride, self.padding, count_include_pad=True,
                            divisor_override=1)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """The pool law on ``x`` (fp64 out); ``x`` is read on its grid within its own dtype's rounding."""
        step = torch.tensor(self.input_step, dtype=torch.float64, device=x.device)
        offset = None if self.input_offset == 0.0 else torch.tensor(self.input_offset, dtype=torch.float64,
                                                                     device=x.device)
        acc = self._sums(grid_codes(x, step, where="pool lane", offset=offset))
        per = {k: torch.as_tensor(v, dtype=torch.float64, device=x.device).reshape(1, -1, 1, 1)
               for k, v in self.law.items()}
        bias = per["bias"]
        if offset is not None:
            bias = bias - per["offset_multiplier"] * self._sums(torch.ones_like(x[:1], dtype=torch.float64))
        codes = apply_law(acc, m0=per["m0"], bias=bias, shift=per["shift"], lo=per["lo"], hi=per["hi"],
                          adc_shift=per["adc_shift"], adc_levels=per["adc_levels"])
        return codes * per["output_step"] - per["output_offset"]


class LanePoolModule(nn.Module):
    """A pool lane's end: STE onto its own lane grid (``quantizer``), or, once stamped, the pool law."""

    def __init__(self, pool: nn.Module, quantizer: nn.Module) -> None:
        super().__init__()
        self.pool = pool
        self.quantizer = quantizer
        self.stamp: Optional[PoolStamp] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)
        if self.stamp is None:
            return self.quantizer(pooled)
        return self.stamp.evaluate(x).to(pooled.dtype) + (pooled - pooled.detach())
