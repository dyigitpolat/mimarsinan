"""Bake TTFS QuantizeDecorator shift into perceptron bias before IR mapping."""

from __future__ import annotations

from mimarsinan.tuning.shift_calculation import calculate_activation_shift
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


def apply_ttfs_quantization_bias_compensation(model, target_tq: int) -> None:
    """Idempotent bias bake when activation quantization is enabled (ttfs_quantized)."""
    for perceptron in model.get_perceptrons():
        if getattr(perceptron, "is_encoding_layer", False):
            continue
        if getattr(perceptron, "_ttfs_shift_baked_into_bias", False):
            continue
        shift = calculate_activation_shift(target_tq, perceptron.activation_scale)
        bias_shift = shift / perceptron.activation_scale
        PerceptronTransformer().apply_effective_bias_transform(
            perceptron, lambda b, s=bias_shift: b + s)
        perceptron._ttfs_shift_baked_into_bias = True


def apply_ttfs_quantized_bias_shift(model, target_tq: int) -> None:
    """Backward-compatible alias for :func:`apply_ttfs_quantization_bias_compensation`."""
    apply_ttfs_quantization_bias_compensation(model, target_tq)
