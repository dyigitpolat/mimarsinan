"""[S3] Sequential first-moment fold for synchronized-TTFS deployments (own-offset-excluded)."""

from __future__ import annotations

from mimarsinan.mapping.support.bias_compensation import SYNC_ENTRY_HALF_STEP_FLAG
from mimarsinan.spiking.dfq_bias_correction import sequential_first_moment_fold
from mimarsinan.spiking.gain_correction import per_perceptron_cascade_depth
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

SYNC_FIRST_MOMENT_FLAG = "_sync_first_moment_folded"
"""Perceptron marker: the [S3] first-moment fold landed on this hop (fold once)."""


def sync_half_step_own_offsets(model, simulation_steps: int) -> dict:
    """Value-domain +θ/(2S) for hops carrying the baked half-step flag.

    The raw deployed-vs-float pre-activation mean gap ≈ +θ/(2S) IS the
    intentional mid-tread compensation; excluding it is load-bearing
    (sync_deployment_exactness.md §3.2: folding it away measured 0.93→0.59)."""
    offsets: dict = {}
    for k, perceptron in enumerate(model.get_perceptrons()):
        if getattr(perceptron, SYNC_ENTRY_HALF_STEP_FLAG, False):
            offsets[k] = calculate_activation_shift(
                int(simulation_steps), perceptron.activation_scale,
            )
    return offsets


def perceptron_forward_order(model) -> list:
    """Cascade-depth (input→output) hop order; declaration order for models
    without a mapper repr (the plain unit-test chains)."""
    perceptrons = list(model.get_perceptrons())
    get_repr = getattr(model, "get_mapper_repr", None)
    if get_repr is None:
        return list(range(len(perceptrons)))
    depths = per_perceptron_cascade_depth(get_repr())
    return sorted(
        range(len(perceptrons)),
        key=lambda k: (int(depths.get(id(perceptrons[k]), 0)), k),
    )


def apply_sync_first_moment_fold(
    model, cal_x, float_preact_mean: dict, simulation_steps: int,
) -> dict:
    """One sequential own-offset-excluded fold pass over the deployed model.

    Run at the sync AQ conversion endpoint (staircase installed at rate 1.0,
    half-step folded), BEFORE endpoint recovery — the QAT then trains from the
    corrected state. Idempotent per hop via ``SYNC_FIRST_MOMENT_FLAG``."""
    return sequential_first_moment_fold(
        model,
        float_preact_mean,
        cal_x,
        own_offsets=sync_half_step_own_offsets(model, simulation_steps),
        hop_order=perceptron_forward_order(model),
        baked_flag=SYNC_FIRST_MOMENT_FLAG,
    )
