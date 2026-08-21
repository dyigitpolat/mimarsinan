"""Deployed-bias compensation: additive effective-bias shifts reconciling training-time vs deployed activation conventions (negative-value shift, TTFS-quantized half-step)."""

from __future__ import annotations

from mimarsinan.mapping.support.negative_shift import (
    apply_negative_shift_bias as apply_negative_shift_bias,
    apply_negative_value_shifts as apply_negative_value_shifts,
    negative_shifts_from_min as negative_shifts_from_min,
    propagate_negative_shifts_to_hybrid as propagate_negative_shifts_to_hybrid,
    transfer_negative_shifts_to_ir as transfer_negative_shifts_to_ir,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

TTFS_COMP_BAKED_FLAG = "_ttfs_shift_baked_into_bias"
"""Perceptron marker: the TTFS half-step compensation is baked into its bias."""


def apply_additive_effective_bias_shift(perceptron, shift, *, baked_flag: str) -> bool:
    """Idempotently add ``shift`` to the effective bias; True when baked by this call."""
    if getattr(perceptron, baked_flag, False):
        return False
    PerceptronTransformer().apply_effective_bias_transform(
        perceptron, lambda b, s=shift: b + s,
    )
    setattr(perceptron, baked_flag, True)
    return True


def apply_ttfs_quantization_bias_compensation(model, target_tq: int) -> None:
    """Idempotent bias bake when activation quantization is enabled (ttfs_quantized)."""
    for perceptron in model.get_perceptrons():
        if getattr(perceptron, "is_encoding_layer", False):
            continue
        shift = calculate_activation_shift(target_tq, perceptron.activation_scale)
        apply_additive_effective_bias_shift(
            perceptron,
            shift / perceptron.activation_scale,
            baked_flag=TTFS_COMP_BAKED_FLAG,
        )


LIF_HALF_STEP_FLAG = "_lif_half_step_baked_into_bias"
SYNC_ENTRY_HALF_STEP_FLAG = "_sync_entry_half_step_folded"


def apply_half_step_entry_fold(
    model,
    simulation_steps: int,
    *,
    baked_flag: str,
    encoders_run_staircase: bool = False,
) -> int:
    """Fold the +theta/(2T) deployed-convention half-step as a TRAINABLE entry
    bias (identical bake math as the TTFS compensation; the QAT owns and may
    train the fold): +theta/(2T) per cycle turns the floor rate grid into
    nearest over the window and head-starts every hop's first fire. Encoders
    fold only when ``encoders_run_staircase`` (their deployed form runs the
    staircase kernel); idempotent per ``baked_flag``; returns folds applied.
    """
    folded = 0
    for perceptron in model.get_perceptrons():
        if not encoders_run_staircase and getattr(perceptron, "is_encoding_layer", False):
            continue
        shift = calculate_activation_shift(
            simulation_steps, perceptron.activation_scale
        )
        if apply_additive_effective_bias_shift(
            perceptron,
            shift / perceptron.activation_scale,
            baked_flag=baked_flag,
        ):
            folded += 1
    return folded


def apply_lif_half_step_bias_compensation(model, simulation_steps: int) -> int:
    """[5v B3] The LIF half-step entry fold: injected before the weight-quant QAT
    so the QAT reconciles the shifted operating point (float NF <-> quantized
    deployed stay bit-exact). Idempotent; returns folds applied."""
    return apply_half_step_entry_fold(
        model, simulation_steps, baked_flag=LIF_HALF_STEP_FLAG,
    )


def apply_sync_exact_entry_half_step(
    model, simulation_steps: int, *, encoding_layer_placement: str,
) -> int:
    """[5v B1 + E1] The sync half-step entry fold before the exact-ceil QAT.

    Placement-aware: a SUBSUMED encoder deploys as a ceil-staircase hop (the
    host ComputeOp runs the perceptron module itself) and receives the fold;
    an OFFLOADED encoder is host-encoded by the mid-tread round and keeps the
    skip. Idempotent; returns folds applied."""
    # Lazy: the torch_mapping package init pulls the converter stack (mapping <-> torch_mapping cycle).
    from mimarsinan.torch_mapping.encoding_layers import encoder_deploys_as_staircase_hop

    return apply_half_step_entry_fold(
        model,
        simulation_steps,
        baked_flag=SYNC_ENTRY_HALF_STEP_FLAG,
        encoders_run_staircase=encoder_deploys_as_staircase_hop(
            encoding_layer_placement
        ),
    )


def _ttfs_segment_calibration_forward(model, x, T, *, compute_min_recorder=None):
    """ttfs_cycle_based boundary values via the genuine single-spike NF driver."""
    from mimarsinan.spiking.segment_forward import SegmentForwardDriver, TtfsSegmentPolicy

    driver = SegmentForwardDriver(model.get_mapper_repr(), T, TtfsSegmentPolicy())
    return driver(x, compute_min_recorder=compute_min_recorder)


def _analytical_segment_calibration_forward(model, x, T, *, compute_min_recorder=None):
    """ttfs / ttfs_quantized boundary values via the pointwise-analytical NF driver."""
    from mimarsinan.spiking.segment_forward import (
        AnalyticalSegmentPolicy,
        SegmentForwardDriver,
    )

    driver = SegmentForwardDriver(model.get_mapper_repr(), T, AnalyticalSegmentPolicy())
    return driver(x, compute_min_recorder=compute_min_recorder)


_NEG_SHIFT_SUPPORTED_MODES = frozenset(
    {"lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"}
)


def calibration_forward_for_mode(spiking_mode: str, *, soma_law):
    """NF forward that produces ``spiking_mode``'s boundary values for calibration.

    The shift must live in the same domain the mode's encoder clamps, so each
    mode calibrates through its own NF forward (resolved by the mode policy),
    carrying the resolved soma point — this walk is the deployed twin."""
    if spiking_mode not in _NEG_SHIFT_SUPPORTED_MODES:
        raise NotImplementedError(
            f"negative_value_shift is not implemented for spiking_mode={spiking_mode!r}"
        )
    from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode

    return policy_for_spiking_mode(
        spiking_mode, soma_law=soma_law).calibration_forward()


