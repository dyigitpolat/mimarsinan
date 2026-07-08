"""Distribution-matching calibration for the genuine TTFS single-spike cascade."""

from __future__ import annotations

import torch

from mimarsinan.spiking.dfq_bias_correction import (
    dfq_correct_biases,
    mean_abs_gap,
    perceptron_channel_mean as _perceptron_channel_mean,
    teacher_activation_samples as _teacher_activation_samples,
)
from mimarsinan.spiking.scale_aware_boundaries import (
    calibrate_scale_aware_boundaries,
    propagate_boundary_input_scales,
)

FANIN_TRAFFIC_QUANTILE = 0.999


def calibrate_fanin_boundary_scales(
    model, cal_x, T, *, quantile: float = FANIN_TRAFFIC_QUANTILE,
    input_data_scale: float,
):
    """Lift fan-in joins' boundary scales from observed traffic (§6b contract-1).

    At a multi-source join the traffic range is up to the SUM of the source
    ranges, so mean-of-producer-θ under-covers and saturates the re-encode
    clamp. One calibration batch through the mode's genuine forward measures
    each join's actual traffic; a join whose q-quantile exceeds its propagated
    out-scale gets a durable ``boundary_traffic_scale`` the JOIN presents to
    BOTH scale walks (wire normalizer and weight fold — one object, so NF↔SCM
    parity holds by construction). In-range traffic changes nothing.
    """
    # Lazy: segment_policies pulls chip_simulation (circular at module top).
    from mimarsinan.spiking.scale_aware_boundaries import read_boundary_out_scales
    from mimarsinan.spiking.segment_forward import (
        SegmentForwardDriver,
        TtfsSegmentPolicy,
    )

    driver = SegmentForwardDriver(model.get_mapper_repr(), int(T), TtfsSegmentPolicy())
    joins: dict = {}
    with torch.no_grad():
        driver(cal_x, join_value_recorder=joins)

    out_scales = read_boundary_out_scales(model, input_data_scale=input_data_scale)
    n_lifted = 0
    for join_node, traffic in joins.items():
        observed = float(
            torch.quantile(traffic.abs().to(torch.float32).flatten(), float(quantile))
        )
        current = float(out_scales.get(join_node, input_data_scale))
        if observed > current:
            join_node.boundary_traffic_scale = observed
            n_lifted += 1

    if n_lifted:
        propagate_boundary_input_scales(model, input_data_scale=input_data_scale)
    return {
        "n_joins": len(joins),
        "n_lifted": n_lifted,
        "quantile": float(quantile),
    }
from mimarsinan.spiking.segment_partition import perceptron_of


def node_values_by_perceptron_index(model, node_values) -> dict:
    """Re-key a ``{mapper_node: value}`` recording by perceptron index."""
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
    return node_values_by_perceptron_index(model, node_values)


def _mean_abs_gap(model, ann_mean, cal_x, T):
    return mean_abs_gap(
        model.get_perceptrons(), ann_mean, _cascade_channel_means(model, cal_x, T),
    )


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
    bias_iters: int,
    input_data_scale: float,
    eta: float = 0.7,
    probe=None,
    probe_patience: int | None = None,
):
    """Match the deployed TTFS cascade's activation distribution to the teacher ANN's.

    Calibrates scale-aware [0,1] boundaries then runs ``bias_iters`` rounds of
    DFQ per-neuron bias correction (keep-best over ``probe`` when given).
    Mutates the model in place (must already be in the deployed
    single-spike-cascade TTFS state); returns a stats dict.
    """
    T = int(T)
    teacher_perceptrons = list(teacher.get_perceptrons())
    n_perceptrons = len(list(model.get_perceptrons()))

    ann_samples = _teacher_activation_samples(teacher, cal_x)
    ann_mean = {
        k: _perceptron_channel_mean(teacher_perceptrons[k], v)
        for k, v in ann_samples.items()
    }
    theta_out = [
        max(float(torch.quantile(ann_samples[k].abs().float().flatten(), quantile)), 1e-2)
        if k in ann_samples else 1.0
        for k in range(n_perceptrons)
    ]

    calibrate_scale_aware_boundaries(
        model, theta_out, input_data_scale=input_data_scale
    )
    fanin_stats = calibrate_fanin_boundary_scales(
        model, cal_x, T, input_data_scale=input_data_scale
    )

    dead_before = _dead_fraction(model, cal_x, T)
    gap_stats = dfq_correct_biases(
        model,
        ann_mean,
        lambda: _cascade_channel_means(model, cal_x, T),
        bias_iters=bias_iters,
        eta=eta,
        probe=probe,
        probe_patience=probe_patience,
    )
    dead_after = _dead_fraction(model, cal_x, T)

    return {
        **gap_stats,
        "dead_fraction_before": sum(dead_before.values()),
        "dead_fraction_after": sum(dead_after.values()),
        "quantile": float(quantile),
        "bias_iters": int(bias_iters),
        "eta": float(eta),
        "num_perceptrons": n_perceptrons,
        "fanin_joins": int(fanin_stats["n_joins"]),
        "fanin_lifted": int(fanin_stats["n_lifted"]),
    }
