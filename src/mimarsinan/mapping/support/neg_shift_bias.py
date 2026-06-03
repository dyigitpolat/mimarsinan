"""Bake a positive-domain input shift into a consuming perceptron's bias.

A ComputeOp that emits negative values is shifted by a per-channel ``s`` so the
spike encoder (rates clamped to [0, 1]) is lossless: it sees ``F(x) + s ≥ 0``.
The consuming core then computes ``W·(F(x)+s) + B``; baking ``B' = B − W·s`` makes
this identical to the unshifted ``W·F(x) + B``. Structurally mirrors
``mimarsinan.mapping.support.ttfs_bias`` (a pre-mapping effective-bias transform).
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


def negative_shifts_from_min(min_by_node: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-node positive shift ``s = max(0, −min F(x))`` so ``F(x)+s ≥ 0``.

    Channels whose observed minimum is already ≥ 0 get shift 0. Nodes whose entire
    shift is 0 are dropped (no shift needed → keeps ``node_output_shifts`` sparse).
    """
    shifts: dict[int, np.ndarray] = {}
    for node_id, mins in min_by_node.items():
        s = np.clip(-np.asarray(mins, dtype=np.float64), a_min=0.0, a_max=None)
        if np.any(s > 0.0):
            shifts[int(node_id)] = s
    return shifts


def _segment_position_shift(stage, node_output_shifts) -> "np.ndarray | None":
    """Map this segment's input-buffer positions to their per-channel shift."""
    total = max((s.offset + s.size for s in stage.input_map), default=0)
    if total == 0:
        return None
    pos_shift = np.zeros(total, dtype=np.float64)
    touched = False
    for sl in stage.input_map:
        s = node_output_shifts.get(int(sl.node_id))
        if s is not None:
            pos_shift[sl.offset : sl.offset + sl.size] = np.asarray(s, dtype=np.float64)[: sl.size]
            touched = True
    return pos_shift if touched else None


def bake_segment_input_shift_bias(stage, node_output_shifts) -> None:
    """Post-mapping: subtract ``W·s`` from each consumer core's per-cycle bias so
    ``W·(input+s)+B' ≡ W·input+B`` on chip (``s`` is the shift on the input the core's
    is-input axons read). Targets ``hardware_bias`` or the always-on (bias) axon row."""
    seg = stage.hard_core_mapping
    if seg is None or not node_output_shifts:
        return
    pos_shift = _segment_position_shift(stage, node_output_shifts)
    if pos_shift is None:
        return
    total = pos_shift.shape[0]

    for core in seg.cores:
        cm = np.asarray(core.core_matrix, dtype=np.float64)
        correction = np.zeros(cm.shape[1], dtype=np.float64)
        always_on_axon = None
        for a, src in enumerate(core.axon_sources):
            if getattr(src, "is_always_on_", False):
                always_on_axon = a
            if getattr(src, "is_input_", False):
                pos = int(src.neuron_)
                sh = pos_shift[pos] if pos < total else 0.0
                if sh != 0.0:
                    correction += cm[a, :] * sh
        if not np.any(correction):
            continue

        hw = getattr(core, "hardware_bias", None)
        if hw is not None:
            core.hardware_bias = np.asarray(hw, dtype=np.float64) - correction
        elif always_on_axon is not None:
            cm = cm.copy()
            cm[always_on_axon, :] -= correction
            core.core_matrix = cm
            core._axon_source_spans = None
        else:
            raise ValueError(
                f"bake_segment_input_shift_bias: core {getattr(core, 'id', '?')} has no "
                "bias (hardware_bias / always-on axon) to absorb the input shift."
            )


def apply_negative_value_shifts(flow, calibration_x) -> dict[int, np.ndarray]:
    """End-to-end Round-2a: calibrate boundary minima on ``calibration_x``, derive
    positive-domain shifts, install them on the hybrid mapping, and bake each
    consumer core's bias. Returns the installed ``node_output_shifts`` (empty if no
    boundary goes negative). The flow's segment-tensor cache is invalidated so the
    new biases take effect on the next forward."""
    mins = flow.calibrate_segment_input_mins(calibration_x)
    shifts = negative_shifts_from_min(mins)
    if not shifts:
        return {}
    flow.hybrid_mapping.node_output_shifts = shifts
    for stage in flow.hybrid_mapping.stages:
        if stage.kind == "neural":
            bake_segment_input_shift_bias(stage, shifts)
    flow._segment_tensor_cache.clear()
    flow._segment_tensor_cache_key = None
    return shifts


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
