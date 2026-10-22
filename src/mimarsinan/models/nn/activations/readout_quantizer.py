"""The readout quantizer on a perceptron's output: STE onto its grid, then the stamped integer law (CH-C1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.integer_law import IntegerLawRangeError, apply_law, grid_codes
from mimarsinan.models.nn.activations.value_quantizer import quantize_to_value_grid

#: The per-channel arrays of a stamped law (the ``CoreLaw`` fields read by the accumulator law).
LAW_FIELDS = ("input_step", "m0", "shift", "bias", "lo", "hi", "output_step", "adc_shift", "adc_levels")


class ReadoutState:
    """One forward's hand-off from the input tap to the quantizer, and the re-entry flag of its rerun."""

    def __init__(self) -> None:
        self.pending: Optional[torch.Tensor] = None
        self.rerunning = False


class ReadoutInputTap(nn.Module):
    """Appended to the perceptron's input wire: hands the (snapped) input to its readout quantizer."""

    def __init__(self, state: ReadoutState) -> None:
        super().__init__()
        self._state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._state.rerunning:
            self._state.pending = x
        return x


@dataclass(eq=False)
class ReadoutStamp:
    """A perceptron's solved law per output channel, its integer weight grid and the input grid step."""

    owner: Tuple[nn.Module, nn.Module]
    weight_name: str
    bias_name: Optional[str]
    weight_grid: np.ndarray
    input_step: float
    law: Dict[str, np.ndarray]
    channel_axis: int
    _cache: Dict[Any, Any] = field(default_factory=dict, repr=False)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    def _on(self, device: torch.device):
        hit = self._cache.get(str(device))
        if hit is None:
            weight = torch.as_tensor(self.weight_grid, dtype=torch.float64, device=device)
            law = {k: torch.as_tensor(v, dtype=torch.float64, device=device) for k, v in self.law.items()}
            hit = (weight, law)
            self._cache[str(device)] = hit
        return hit

    def accumulate(self, codes: torch.Tensor, state: ReadoutState) -> torch.Tensor:
        """The integer accumulator: the owner's own forward on the input codes, integer weights, zero bias."""
        mapper, perceptron = self.owner
        weight, _law = self._on(codes.device)
        overrides = {self.weight_name: weight}
        if self.bias_name is not None:
            overrides[self.bias_name] = weight.new_zeros(weight.shape[0])
        saved = (perceptron.input_activation, perceptron.activation)
        perceptron.input_activation, perceptron.activation = nn.Identity(), nn.Identity()
        state.rerunning = True
        try:
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
                acc = torch.func.functional_call(mapper, overrides, (codes,))
        finally:
            perceptron.input_activation, perceptron.activation = saved
            state.rerunning = False
        if not bool(torch.equal(acc, torch.round(acc))):
            raise IntegerLawRangeError(f"{type(mapper).__name__}: the fp64 accumulator is not integral")
        return acc

    def evaluate(self, x: torch.Tensor, state: ReadoutState) -> torch.Tensor:
        """The law's output values ``y Δout`` (fp64) for the perceptron input ``x``."""
        step = torch.tensor(self.input_step, dtype=torch.float64, device=x.device)
        acc = self.accumulate(grid_codes(x, step, where=type(self.owner[0]).__name__), state)
        _weight, law = self._on(acc.device)
        shape = [1] * acc.dim()
        shape[self.channel_axis] = -1
        per = {k: v.reshape(shape) for k, v in law.items()}
        codes = apply_law(acc, m0=per["m0"], bias=per["bias"], shift=per["shift"], lo=per["lo"], hi=per["hi"],
                          adc_shift=per["adc_shift"], adc_levels=per["adc_levels"])
        return codes * per["output_step"]


class ReadoutQuantizer(nn.Module):
    """STE onto the output grid ``±range`` until a law is stamped; then the exact integer law on the
    accumulator the perceptron's own forward computes from integer codes and integer weights."""

    range: torch.Tensor

    def __init__(self, activation_bits: int, range_value: float = 0.0) -> None:
        super().__init__()
        self.activation_bits = int(activation_bits)
        self.register_buffer("range", torch.tensor(float(range_value)))
        self._state = ReadoutState()
        self._stamp: Optional[ReadoutStamp] = None

    def calibrate(self, range_value: float) -> None:
        self.range.fill_(float(range_value))

    def input_tap(self) -> ReadoutInputTap:
        return ReadoutInputTap(self._state)

    def stamp(self, stamp: ReadoutStamp) -> None:
        self._stamp = stamp

    @property
    def stamped(self) -> Optional[ReadoutStamp]:
        return self._stamp

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        state = self._state
        if state.rerunning:
            return y
        if self._stamp is None:
            quantized = quantize_to_value_grid(y, self.range, self.activation_bits)
            return y + (quantized - y).detach()
        x, state.pending = state.pending, None
        if x is None:
            raise RuntimeError("ReadoutQuantizer: no input reached its tap; the law reads the perceptron input")
        values = self._stamp.evaluate(x, state)
        return values.to(y.dtype) + (y - y.detach())
