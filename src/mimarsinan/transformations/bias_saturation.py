"""Saturation-equivalent effective-bias canonicalization for shared-grid quantization."""

from __future__ import annotations

import torch

from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)


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

    def clip(effective_bias):
        nonlocal clipped
        limit = bound.to(effective_bias.device, effective_bias.dtype)
        below = effective_bias < limit
        clipped = int(below.sum())
        return torch.where(below, limit, effective_bias)

    transformer.apply_effective_bias_transform(perceptron, clip)
    if clipped:
        print(
            f"[BiasSaturation] {getattr(perceptron, 'name', '<unnamed>')}: "
            f"clipped {clipped} constant-off channel bias(es) to the "
            "saturation bound"
        )
    return clipped
