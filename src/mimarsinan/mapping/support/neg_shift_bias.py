"""Bake a positive-domain input shift into a consuming perceptron's bias.

A ComputeOp that emits negative values is shifted by a per-channel ``s`` so the
spike encoder (rates clamped to [0, 1]) is lossless: it sees ``F(x) + s ≥ 0``.
The consuming core then computes ``W·(F(x)+s) + B``; baking ``B' = B − W·s`` makes
this identical to the unshifted ``W·F(x) + B``. Structurally mirrors
``mimarsinan.mapping.support.ttfs_bias`` (a pre-mapping effective-bias transform).
"""

from __future__ import annotations

import torch

from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


def apply_negative_shift_bias(perceptron, shift) -> None:
    """Idempotently bake ``B' = B − W·s`` (``s`` per-axon or scalar) into ``perceptron``."""
    if getattr(perceptron, "_neg_shift_baked", False):
        return

    transformer = PerceptronTransformer()
    effective_weight = transformer.get_effective_weight(perceptron)  # (neurons, axons)
    s = torch.as_tensor(
        shift, dtype=effective_weight.dtype, device=effective_weight.device,
    )
    # Per neuron j: Σ_axon W_eff[j, axon] · s[axon]. Scalar s broadcasts over axons.
    correction = (effective_weight * s).sum(dim=-1)

    transformer.apply_effective_bias_transform(
        perceptron, lambda b, c=correction: b - c,
    )
    perceptron._neg_shift_baked = True
