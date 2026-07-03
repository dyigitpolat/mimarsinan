"""Distribution-matching calibration for the genuine TTFS single-spike cascade."""

from __future__ import annotations

import torch

from mimarsinan.spiking.dfq_bias_correction import (
    channel_mean as _channel_mean,
    dfq_correct_biases,
    mean_abs_gap,
    teacher_activation_samples as _teacher_activation_samples,
)
from mimarsinan.spiking.scale_aware_boundaries import calibrate_scale_aware_boundaries
from mimarsinan.spiking.segment_partition import perceptron_of


def _cascade_channel_means(model, cal_x, T):
    """Per-perceptron cascade decoded value, keyed by perceptron index."""
    # Lazy: ttfs_segment_forward imports mimarsinan.spiking, so a top-level import is circular.
    from mimarsinan.models.spiking.training.ttfs_segment_forward import (
        TTFSSegmentForward,
    )

    with torch.no_grad():
        _, node_values = TTFSSegmentForward(
            model.get_mapper_repr(), T
        ).forward_with_node_values(cal_x)
    by_perceptron = {
        id(perceptron_of(node)): value
        for node, value in node_values.items()
        if perceptron_of(node) is not None
    }
    out: dict = {}
    for k, perceptron in enumerate(model.get_perceptrons()):
        value = by_perceptron.get(id(perceptron))
        if value is not None:
            out[k] = value
    return out


def _mean_abs_gap(model, ann_mean, cal_x, T):
    return mean_abs_gap(ann_mean, _cascade_channel_means(model, cal_x, T))


def _dead_fraction(model, cal_x, T):
    cascade = _cascade_channel_means(model, cal_x, T)
    return {
        k: (value.abs() < 1e-6).float().mean().item()
        for k, value in cascade.items()
    }


def match_activation_distributions(
    model,
    teacher,
    cal_x,
    T,
    *,
    quantile: float = 0.99,
    bias_iters: int = 15,
    eta: float = 0.7,
):
    """Match the deployed TTFS cascade's activation distribution to the teacher ANN's.

    Calibrates scale-aware [0,1] boundaries then runs ``bias_iters`` rounds of
    DFQ per-neuron bias correction. Mutates the model in place (must already be
    in the deployed single-spike-cascade TTFS state); returns a stats dict.
    """
    T = int(T)
    n_perceptrons = len(list(model.get_perceptrons()))

    ann_samples = _teacher_activation_samples(teacher, cal_x)
    ann_mean = {k: _channel_mean(v) for k, v in ann_samples.items()}
    theta_out = [
        max(float(torch.quantile(ann_samples[k].abs().float().flatten(), quantile)), 1e-2)
        if k in ann_samples else 1.0
        for k in range(n_perceptrons)
    ]

    calibrate_scale_aware_boundaries(model, theta_out)

    dead_before = _dead_fraction(model, cal_x, T)
    gap_stats = dfq_correct_biases(
        model,
        ann_mean,
        lambda: _cascade_channel_means(model, cal_x, T),
        bias_iters=bias_iters,
        eta=eta,
    )
    dead_after = _dead_fraction(model, cal_x, T)

    return {
        "mean_gap_before": gap_stats["mean_gap_before"],
        "mean_gap_after": gap_stats["mean_gap_after"],
        "dead_fraction_before": sum(dead_before.values()),
        "dead_fraction_after": sum(dead_after.values()),
        "quantile": float(quantile),
        "bias_iters": int(bias_iters),
        "eta": float(eta),
        "num_perceptrons": n_perceptrons,
    }
