"""Saturation-equivalent effective-bias canonicalization for shared-grid quantization."""

from __future__ import annotations

import torch

from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)

_STARVATION_ZERO_FRAC = 0.8
"""Predicted weight-zero fraction at the target grid above which a perceptron
is starved enough to arm the guarded empirical pass."""

_EMPIRICAL_MARGIN = 0.25
"""Saturation slack (in ceiling units) the empirical bias shift must keep."""


def off_saturation_bias_bound(effective_weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel bound ``-pos_reach``: the largest effective bias at
    which a floor-0 channel stays constant-OFF for every input in ``[0, 1]``."""
    w = (
        effective_weight.flatten(1)
        if effective_weight.dim() > 1
        else effective_weight.unsqueeze(1)
    )
    return -w.clamp(min=0).sum(dim=1)


def clip_off_saturated_effective_bias(perceptron) -> int:
    """Clip constant-OFF channels' effective bias to the saturation bound.

    A channel with ``b + pos_reach <= 0`` outputs the activation floor (0) for
    every normalized input in [0, 1]; bias mass below ``-pos_reach`` is
    functionally unobservable, yet it participates in the shared per-perceptron
    weight/bias quantization scale and can starve the whole weight grid (the
    t01_19/t0_03 WQ-entry crater: one dead-channel b_eff = -12.6 zeroed 100%
    of a layer's 4-bit weights). Function-preserving and idempotent; encoding
    layers are skipped (their input domain is not the normalized [0, 1]).
    Returns the number of channels clipped.
    """
    if getattr(perceptron, "is_encoding_layer", False):
        return 0
    transformer = PerceptronTransformer()
    bound = off_saturation_bias_bound(
        transformer.get_effective_weight(perceptron).detach()
    )
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

    transformer.apply_effective_bias_transform(perceptron, clip)
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
    """Fraction of effective weights that round to zero on the shared grid."""
    transformer = PerceptronTransformer()
    w = transformer.get_effective_weight(perceptron).detach().abs()
    b = transformer.get_effective_bias(perceptron).detach().abs()
    q_max = (2 ** (bits - 1)) - 1
    p_max = max(float(w.max()), float(b.max()), 1e-12)
    step = p_max / q_max
    return float((w < step / 2).float().mean())
