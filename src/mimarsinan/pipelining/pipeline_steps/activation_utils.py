"""Shared helpers for activation adaptation steps."""

import torch

RELU_COMPATIBLE_TYPES = (
    "LeakyGradReLU",  # forward is pure ReLU; leaky only in backward
    "ReLU",
)


def activation_scale_stats(
    perceptrons,
    sampled_activations,
    activation_scales,
    *,
    num_batches,
    quantile,
    max_samples_per_batch,
    pruned_threshold,
):
    """Per-layer activation-scale calibration stats (the analysis-step report)."""
    layer_stats = []
    for idx, (perceptron, samples, scale) in enumerate(
        zip(perceptrons, sampled_activations, activation_scales)
    ):
        sample_count = int(samples.numel())
        active_samples = samples[samples > pruned_threshold]
        if sample_count > 0:
            sample_min = float(samples.min().item())
            sample_median = float(torch.quantile(samples, 0.5).item())
            sample_max = float(samples.max().item())
        else:
            sample_min = 0.0
            sample_median = 0.0
            sample_max = 0.0

        layer_stats.append(
            {
                "index": idx,
                "name": perceptron.name,
                "scale": float(scale),
                "sample_count": sample_count,
                "active_sample_count": int(active_samples.numel()),
                "sample_min": sample_min,
                "sample_median": sample_median,
                "sample_max": sample_max,
            }
        )

    sorted_scales = sorted(float(s) for s in activation_scales) or [1.0]
    return {
        "num_batches": int(num_batches),
        "quantile": float(quantile),
        "pruned_threshold": float(pruned_threshold),
        "max_samples_per_batch": int(max_samples_per_batch),
        "summary": {
            "min_scale": sorted_scales[0],
            "median_scale": sorted_scales[len(sorted_scales) // 2],
            "max_scale": sorted_scales[-1],
        },
        "layers": layer_stats,
    }


def needs_relu_adaptation(perceptron) -> bool:
    """True when a perceptron's base activation is not already ReLU-compatible."""
    base_name = type(perceptron.base_activation).__name__
    return base_name not in RELU_COMPATIBLE_TYPES


def has_non_relu_activations(model) -> bool:
    """True if any chip-targeted perceptron has a non-ReLU base (e.g. GELU, LeakyReLU)."""
    for p in model.get_perceptrons():
        if needs_relu_adaptation(p):
            return True
    return False
