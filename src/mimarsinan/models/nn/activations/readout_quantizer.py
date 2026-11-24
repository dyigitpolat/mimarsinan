"""The readout quantizer on a perceptron's output: STE onto its grid, then the stamped integer law (CH-C1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.nn.activations.integer_law import IntegerLawRangeError, apply_law, grid_codes, round_half_away
from mimarsinan.models.nn.activations.lane_quantization import LaneSeam, quantize_to_lane_grid
from mimarsinan.models.nn.activations.value_quantizer import boundary_grid_of, quantize_to_value_grid

#: The per-channel arrays of a stamped law (the ``CoreLaw`` fields read by the accumulator law).
LAW_FIELDS = ("input_step", "m0", "shift", "bias", "lo", "hi", "output_step", "adc_shift", "adc_levels")
#: The stamped-law key of the per-channel offset multiplier ``M_o`` of an offset entry grid (WI-5a; else absent).
OFFSET_MULTIPLIER = "offset_multiplier"
#: The stamped-law keys of a lane (WI-5e; else absent): the output grid's offset and the fused operand's multiplier,
#: step and offset per channel.
OUTPUT_OFFSET, OPERAND_M, OPERAND_STEP, OPERAND_OFFSET = "output_offset", "operand_m", "operand_step", "operand_offset"


def _operand_steps(operand: torch.Tensor, steps: torch.Tensor, owner: str) -> torch.Tensor:
    """The fused operand's grid step per channel, an infinite step (code 0) where the law reads no operand: a
    channel whose operand is an off element (a zero-padded or pruned one, step 0 and multiplier 0 in the law) adds
    0, as its off operand row does on the cores. A nonzero value there is refused by name."""
    none = steps == 0
    if not bool(none.any()):
        return steps
    if bool((torch.where(none, operand.to(torch.float64), torch.zeros((), dtype=torch.float64)) != 0).any()):
        raise IntegerLawRangeError(f"{owner} operand: a nonzero value in a channel whose law reads no operand (an off "
                                   "element adds 0)")
    return torch.where(none, torch.full_like(steps, float("inf")), steps)


def straight_through(values: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The stamped law's estimator: the law's ``values`` forward (in ``y``'s dtype), the identity onto the
    perceptron's float output ``y`` backward."""
    return values.to(y.dtype) + (y - y.detach())


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

    def _input_offset(self, device: torch.device) -> Optional[torch.Tensor]:
        """The offset of the perceptron's input grid (``None`` without one): its codes read ``x + o``. A lane stamp
        names it (a core-fed perceptron reads its producer's offset grid); else the declared entry grid's."""
        if "offset" not in self._cache:
            lane = getattr(self, "lane_input_offset", None)
            grid = boundary_grid_of(getattr(self.owner[1], "input_activation", None))
            value = float(lane) if lane is not None else 0.0 if grid is None else grid.entry_offset()
            self._cache["offset"] = None if value == 0.0 else value
        offset = self._cache["offset"]
        return None if offset is None else torch.tensor(offset, dtype=torch.float64, device=device)

    def evaluate(self, x: torch.Tensor, state: ReadoutState,
                 operand: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The law's output values ``y Δout - o_out`` (fp64) for the perceptron input ``x`` (and a fused add's
        ``operand``, read on its own grid)."""
        step = torch.tensor(self.input_step, dtype=torch.float64, device=x.device)
        offset = self._input_offset(x.device)
        codes = grid_codes(x, step, where=type(self.owner[0]).__name__, offset=offset)
        acc = self.accumulate(codes, state)
        _weight, law = self._on(acc.device)
        reduce = getattr(self, "lane_reduce", None)
        if reduce is not None:
            return self._reduced(acc, law, reduce)
        shape = [1] * acc.dim()
        shape[self.channel_axis] = -1
        per = {k: v.reshape(shape) for k, v in law.items()}
        bias = per["bias"]
        if (offset is None) != (OFFSET_MULTIPLIER not in per):
            raise IntegerLawRangeError(f"{type(self.owner[0]).__name__}: an entry offset and its law disagree")
        if offset is not None:
            # The offset term over the rows the input reaches: the accumulator of ones (zero padding included).
            bias = bias - per[OFFSET_MULTIPLIER] * self.accumulate(torch.ones_like(codes[:1]), state)
        operands = ()
        if (operand is None) != (OPERAND_M not in per):
            raise IntegerLawRangeError(f"{type(self.owner[0]).__name__}: a lane operand and its law disagree")
        if operand is not None:
            q = grid_codes(operand, _operand_steps(operand, per[OPERAND_STEP], type(self.owner[0]).__name__),
                           where=f"{type(self.owner[0]).__name__} operand", offset=per[OPERAND_OFFSET])
            operands = ((q, per[OPERAND_M]),)
        codes = apply_law(acc, m0=per["m0"], bias=bias, shift=per["shift"], lo=per["lo"], hi=per["hi"],
                          adc_shift=per["adc_shift"], adc_levels=per["adc_levels"], operands=operands)
        values = codes * per["output_step"]
        return values - per[OUTPUT_OFFSET] if OUTPUT_OFFSET in per else values

    @staticmethod
    def _reduced(acc: torch.Tensor, law: Dict[str, torch.Tensor], window) -> torch.Tensor:
        """A producer-side reduction (WI-5e): the accumulators summed over each pool window (zero padding adds
        nothing), then the law per pool output element, whose bias the IR folded per element (instances present,
        input offset)."""
        kernel, stride, padding = window
        summed = F.avg_pool2d(acc, kernel, stride, padding, count_include_pad=True, divisor_override=1)
        per = {k: v.reshape((1,) + tuple(summed.shape[1:])) for k, v in law.items()}
        codes = apply_law(summed, m0=per["m0"], bias=per["bias"], shift=per["shift"], lo=per["lo"], hi=per["hi"],
                          adc_shift=per["adc_shift"], adc_levels=per["adc_levels"])
        return codes * per["output_step"] - per[OUTPUT_OFFSET]


def snap_channel_groups(y: torch.Tensor, ranges: torch.Tensor, levels: torch.Tensor, axis: int) -> torch.Tensor:
    """``y`` snapped per channel onto its group's grid ``±range`` of ``levels`` positive levels (ties away)."""
    shape = [1] * y.dim()
    shape[axis] = -1
    ranges = ranges.to(y.dtype).reshape(shape)
    levels = levels.to(y.dtype).reshape(shape)
    step = torch.where(ranges > 0, ranges / levels, torch.ones_like(ranges))
    snapped = torch.clamp(round_half_away(y / step), -levels, levels) * step
    return torch.where(ranges > 0, snapped, y)


class ReadoutQuantizer(nn.Module):
    """STE onto the output grid ``±range`` until a law is stamped; then the exact integer law on the
    accumulator the perceptron's own forward computes from integer codes and integer weights. With consumer
    groups (N2, ``readout_ranges: per_consumer_group``) each group of output channels has its own range and width.

    On a lane (WI-5e, :meth:`set_lane`) the grid is unsigned (codes ``0..2^b - 1``, offset ``o``) where the lane
    clamps or declares an offset, and a lane that continues through a fused add rounds at its end
    (:meth:`end_lane`), so its perceptron's own slot passes the value through."""

    range: torch.Tensor
    group_ranges: torch.Tensor
    #: N2 consumer groups (set by ``set_groups``; absent on a per-perceptron quantizer, whose state is unchanged).
    channel_groups: Optional[np.ndarray] = None
    group_levels: Tuple[int, ...] = ()
    channel_axis: int = -1
    #: Lane configuration, set only by :meth:`set_lane` (class defaults: no lane, the pickle unchanged).
    lane_unsigned: bool = False
    lane_offset: float = 0.0
    lane_continues: bool = False

    def __init__(self, activation_bits: int, range_value: float = 0.0) -> None:
        super().__init__()
        self.activation_bits = int(activation_bits)
        self.register_buffer("range", torch.tensor(float(range_value)))
        self._state = ReadoutState()
        self._stamp: Optional[ReadoutStamp] = None

    def calibrate(self, range_value: float) -> None:
        self.range.fill_(float(range_value))
        if self.channel_groups is not None:
            self.group_ranges.fill_(float(range_value))

    def set_groups(self, channels: np.ndarray, levels: Tuple[int, ...], channel_axis: int) -> None:
        """Per output channel its group, per group its positive levels; each group's range starts at 0."""
        self.channel_groups = np.asarray(channels, dtype=np.int64)
        self.group_levels = tuple(int(level) for level in levels)
        self.channel_axis = int(channel_axis)
        self.register_buffer("group_ranges", torch.zeros(len(self.group_levels)))

    def calibrate_groups(self, ranges) -> None:
        self.group_ranges.copy_(torch.as_tensor([float(r) for r in ranges]))
        self.range.fill_(max(float(r) for r in ranges))

    def _snap(self, y: torch.Tensor) -> torch.Tensor:
        if self.channel_groups is None:
            return quantize_to_value_grid(y, self.range, self.activation_bits)
        index = torch.as_tensor(self.channel_groups, device=y.device)
        levels = torch.as_tensor(self.group_levels, device=y.device)[index]
        return snap_channel_groups(y, self.group_ranges.to(y.device)[index], levels, self.channel_axis)

    def input_tap(self) -> ReadoutInputTap:
        return ReadoutInputTap(self._state)

    def stamp(self, stamp: ReadoutStamp) -> None:
        self._stamp = stamp

    @property
    def stamped(self) -> Optional[ReadoutStamp]:
        return self._stamp

    def adopt_law(self, other: "ReadoutQuantizer") -> None:
        """[U-ADC] Read the law ``other`` carries: its stamp and its hand-off state, the state its stamp's accumulator
        rerun flags on ``other``'s own perceptron (a live model training through a law-carrying copy's laws)."""
        self.__dict__.setdefault("_own_state", self._state)
        self._stamp, self._state = other._stamp, other._state

    def release_law(self) -> None:
        """[U-ADC] Back to the unstamped quantizer on its own hand-off state."""
        self._stamp = None
        self._state = self.__dict__.pop("_own_state", self._state)

    def set_lane(self, *, unsigned: bool, offset: float = 0.0, continues: bool = False) -> None:
        """Make this quantizer its perceptron's lane: an unsigned (offset) grid, and its rounding at a fused
        add's end when the lane ``continues`` (the seam its calibration reads is then :attr:`lane_seam`)."""
        if offset != 0.0 and not unsigned:
            raise ValueError("an offset lane grid is unsigned (codes 0..2^b - 1)")
        self.lane_unsigned, self.lane_offset, self.lane_continues = bool(unsigned), float(offset), bool(continues)
        if continues:
            self.lane_seam = LaneSeam()

    def quantize(self, y: torch.Tensor) -> torch.Tensor:
        """The value grid this quantizer writes (no STE): the lane grid on an unsigned lane, else :meth:`_snap`
        (each consumer group on its own range and width; without groups the per-perceptron grid)."""
        if self.lane_unsigned:
            return quantize_to_lane_grid(y, self.range, self.activation_bits, self.lane_offset)
        return self._snap(y)

    def end_lane(self, v: torch.Tensor, operand: Optional[torch.Tensor]) -> torch.Tensor:
        """A fused add's or a reduction's end: STE of its value ``v`` (the sum, the pooled value) onto the lane grid,
        or the stamped law (with a fused add's ``operand``)."""
        if self._stamp is None:
            v = self.lane_seam(v)
            return v + (self.quantize(v) - v).detach()
        x, self._state.pending = self._state.pending, None
        if x is None:
            raise RuntimeError("ReadoutQuantizer: no input reached its tap; the lane law reads the perceptron input")
        values = self._stamp.evaluate(x, self._state, operand=operand)
        return straight_through(values, v)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        state = self._state
        if state.rerunning or self.lane_continues:
            return y
        if self._stamp is None:
            quantized = self.quantize(y)
            return y + (quantized - y).detach()
        x, state.pending = state.pending, None
        if x is None:
            raise RuntimeError("ReadoutQuantizer: no input reached its tap; the law reads the perceptron input")
        values = self._stamp.evaluate(x, state)
        return straight_through(values, y)
