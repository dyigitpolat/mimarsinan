"""Saturation-equivalent effective-bias canonicalization for shared-grid quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from mimarsinan.models.nn.activations.value_quantizer import carries_value_grid, wire_output_grid
from mimarsinan.models.nn.off_floor import has_off_floor
from mimarsinan.models.perceptron_mixer.perceptron import reads_value_domain
from mimarsinan.transformations.perceptron.bias_drive import bias_row_values
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)
from mimarsinan.transformations.perceptron.position_fold import (
    PER_POSITION, bias_offsets, effective_bias_rows, effective_weight_rows, holds_positions, outputs_first,
    position_folds)

_STARVATION_ZERO_FRAC = 0.8
"""Predicted weight-zero fraction at the target grid above which a perceptron
is starved enough to arm the guarded empirical pass."""

_EMPIRICAL_MARGIN = 0.25
"""Saturation slack (in ceiling units) the empirical bias shift must keep."""

_FLOAT32_EPS = 2.0 ** -23
_BOUND_HEADROOM_STEPS = 8
"""Rounding steps beyond the fan-in that a declared-range bound keeps in hand (the bias add and its write-back)."""


@dataclass(frozen=True)
class InputRange:
    """Per-input bounds ``[low, high]`` of a perceptron's effective input, broadcastable against its effective weight."""

    low: Union[torch.Tensor, float]
    high: Union[torch.Tensor, float]


UNIT_INPUT_RANGE = InputRange(0.0, 1.0)
"""The event wire's normalized domain: spike rates in [0, 1] (the negative-boundary policy keeps signed values off chip)."""


def _per_output_channel(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(1) if t.dim() > 1 else t.unsqueeze(1)


def off_saturation_bias_bound(
    effective_weight: torch.Tensor, input_range: InputRange = UNIT_INPUT_RANGE,
) -> torch.Tensor:
    """Per-output-channel bound ``-max_x w.x`` over ``input_range``: the largest
    effective bias at which a floor-0 channel stays constant-OFF for every input there.

    The event wire's [0, 1] keeps the 75998141 formula bit for bit. On a declared range the maximizing
    corner is a grid point, so the bound is taken in float64 and lowered by the float32 error bound of
    the forward's sum, ``(fan_in + 8) eps32 sum_i |w_i| max(|lo_i|, |hi_i|)``: a clipped channel stays
    OFF in the layer's own float32 forward."""
    if input_range is UNIT_INPUT_RANGE:
        return -_per_output_channel(effective_weight).clamp(min=0).sum(dim=1)
    w = effective_weight.double()
    low = torch.as_tensor(input_range.low, dtype=torch.float64, device=w.device)
    high = torch.as_tensor(input_range.high, dtype=torch.float64, device=w.device)
    reach = _per_output_channel(torch.maximum(w * low, w * high))
    magnitude = _per_output_channel(w.abs() * torch.maximum(low.abs(), high.abs())).sum(dim=1)
    margin = (reach.shape[1] + _BOUND_HEADROOM_STEPS) * _FLOAT32_EPS * magnitude
    return (-reach.sum(dim=1) - margin).to(effective_weight.dtype)


def declared_input_range(perceptron, transformer=None) -> Optional[InputRange]:
    """The proven range of the effective input (``x / per_input_scales``): the value range of the
    boundary grid the input wire ends on, hulled with 0 (zero padding); the event wire's [0, 1] for a
    wire with no value-domain grid that is not declared a value-domain reader; None for a
    value-domain reader whose output no armed grid bounds."""
    wire = getattr(perceptron, "input_activation", None)
    if not carries_value_grid(wire):
        return None if reads_value_domain(perceptron) else UNIT_INPUT_RANGE
    grid = wire_output_grid(wire)
    if grid is None:
        return None
    low, high = grid.value_range
    # [U-PFOLD] a fold per position: each position's inputs at its own row (the one reader's, on the weight's device)
    scale = position_folds(perceptron) if holds_positions(perceptron) else (
        transformer or PerceptronTransformer()).input_scale(perceptron)
    a, b = min(low, 0.0) / scale, max(high, 0.0) / scale
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return InputRange(torch.minimum(a, b), torch.maximum(a, b))
    return InputRange(a, b)


def constant_off_input_range(perceptron, transformer=None) -> Optional[InputRange]:
    """The input range on which a channel's constant-OFF state is provable, or None when none is:
    an encoding layer, a node that declares no floor (``models.nn.off_floor``; an affine package's
    output is linear in its bias), or no proven range."""
    if getattr(perceptron, "is_encoding_layer", False):
        return None
    if not has_off_floor(getattr(perceptron, "base_activation", None)):
        return None
    return declared_input_range(perceptron, transformer)


def clip_off_saturated_effective_bias(perceptron, transformer=None) -> int:
    """Clip constant-OFF channels' effective bias to the saturation bound.

    A channel with ``b + max_x w.x <= 0`` outputs the activation floor (0) for
    every input of its proven range (``constant_off_input_range``); bias mass
    below that bound is functionally unobservable, yet it participates in the
    shared per-perceptron weight/bias quantization scale and can starve the
    whole weight grid (the t01_19/t0_03 WQ-entry crater: one dead-channel
    b_eff = -12.6 zeroed 100% of a layer's 4-bit weights). Function-preserving
    and idempotent; a perceptron with no provable OFF state is left alone (D19:
    the value-domain stem reading the signed image). Returns the number of channels clipped.
    """
    if transformer is None:
        transformer = PerceptronTransformer()
    input_range = constant_off_input_range(perceptron, transformer)
    if input_range is None:
        return 0
    bound = _off_bound(perceptron, transformer, input_range)
    clipped = 0
    max_delta = 0.0

    def clip(effective_bias):
        nonlocal clipped, max_delta
        limit = bound.to(effective_bias.device, effective_bias.dtype)
        below = effective_bias < limit
        clipped = int(below.sum())
        if clipped:
            max_delta = float((limit - effective_bias)[below].max())
        return torch.where(below, limit, effective_bias)

    # [U-PFOLD] a fold per position clips the shared bias to the bound every position keeps (its offsets as they are)
    (transformer.write_shared_bias if holds_positions(perceptron) else transformer.apply_effective_bias_transform)(
        perceptron, clip)
    # QAT drift re-clips dead channels by epsilons every projection; only a
    # MATERIAL clip (a real crater canonicalization) is worth a log line.
    if clipped and max_delta > 1e-3:
        print(
            f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
            f"clipped {clipped} constant-off channel bias(es) to the "
            f"saturation bound (max delta {max_delta:.3f})"
        )
    return clipped


def empirical_bias_shift(
    effective_bias: torch.Tensor,
    v_min: torch.Tensor,
    v_max: torch.Tensor,
    *,
    ceiling: float,
    margin: float,
) -> torch.Tensor:
    """Per-channel bias shift toward zero for EMPIRICALLY saturated channels.

    A channel whose observed pre-activation never leaves saturation
    (``v_min >= ceiling`` or ``v_max <= 0``) tolerates a uniform shift that
    keeps ``margin`` of saturation slack. The shift only ever shrinks ``|b|``
    and never crosses zero; unsaturated channels get exactly 0.
    """
    zero = torch.zeros_like(effective_bias)
    on = v_min >= ceiling
    off = v_max <= 0.0
    delta = torch.where(
        on, (ceiling - v_min) + margin,
        torch.where(off, (-v_max) - margin, zero),
    )
    shifted = effective_bias + delta
    shifted = torch.where(
        effective_bias > 0, shifted.clamp(min=0.0), shifted.clamp(max=0.0),
    )
    shrinks = shifted.abs() < effective_bias.abs()
    return torch.where((on | off) & shrinks, shifted - effective_bias, zero)


def predicted_weight_zero_fraction(perceptron, bits: int) -> float:
    """Fraction of effective weights that round to zero on the shared grid (the bias row read at its drive)."""
    transformer = PerceptronTransformer()
    w, b = position_magnitudes(perceptron, transformer)
    q_max = (2 ** (bits - 1)) - 1
    p_max = max(float(w.max()), float(b.max()), 1e-12)
    step = p_max / q_max
    return float((w < step / 2).float().mean())


def _off_bound(perceptron, transformer, input_range: InputRange) -> torch.Tensor:
    """The per-output-channel OFF bound; [U-PFOLD] over a fold per position the minimum over positions of each
    position's bound less its bias offset (the shared bias at it keeps every position OFF)."""
    if not holds_positions(perceptron):
        return off_saturation_bias_bound(transformer.get_effective_weight(perceptron).detach(), input_range)
    rows, index = effective_weight_rows(perceptron)
    weights = outputs_first(rows[index].detach(), PER_POSITION.weight)  # out, positions, fan-in
    out, positions, fan_in = (int(d) for d in weights.shape)

    def per_row(v):
        v = torch.as_tensor(v, dtype=weights.dtype, device=weights.device)
        return v if v.dim() == 0 else v.reshape(1, positions, fan_in).expand(out, positions, fan_in).reshape(-1, fan_in)

    rows_range = input_range if input_range is UNIT_INPUT_RANGE else InputRange(per_row(input_range.low),
                                                                               per_row(input_range.high))
    bound = off_saturation_bias_bound(weights.reshape(-1, fan_in), rows_range).reshape(out, positions)
    offsets = bias_offsets(perceptron)
    if offsets is not None:
        bound = bound - outputs_first(offsets, PER_POSITION.bias).to(bound.dtype)
    return bound.amin(dim=1)


def position_magnitudes(perceptron, transformer) -> tuple[torch.Tensor, torch.Tensor]:
    """``(|w~|, |b~| / drive)``: [U-PFOLD] every position's where the fold or the bias is held per position."""
    if not holds_positions(perceptron):
        return (transformer.get_effective_weight(perceptron).detach().abs(),
                bias_row_values(perceptron, transformer.get_effective_bias(perceptron).detach().abs()))
    rows, index = effective_weight_rows(perceptron)
    shared = transformer.shared_effective_bias(perceptron).detach()
    return rows[index].detach().abs(), bias_row_values(perceptron, effective_bias_rows(perceptron, shared).abs())
