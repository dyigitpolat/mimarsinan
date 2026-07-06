"""Deployed-bias compensation: additive effective-bias shifts reconciling training-time vs deployed activation conventions (negative-value shift, TTFS-quantized half-step)."""

from __future__ import annotations

import numpy as np
import torch

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


def negative_shifts_from_min(min_by_node: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-node positive shift ``s = max(0, −min F(x))``; drops all-zero nodes."""
    shifts: dict[int, np.ndarray] = {}
    for node_id, mins in min_by_node.items():
        s = np.clip(-np.asarray(mins, dtype=np.float64), a_min=0.0, a_max=None)
        if np.any(s > 0.0):
            shifts[int(node_id)] = s
    return shifts


def apply_negative_shift_bias(perceptron, shift) -> None:
    """Idempotently bake ``B' = B − W·s`` (``s`` per-axon or scalar) into ``perceptron``."""
    if getattr(perceptron, "_neg_shift_baked", False):
        return

    effective_weight = PerceptronTransformer().get_effective_weight(perceptron)
    s = torch.as_tensor(
        shift, dtype=effective_weight.dtype, device=effective_weight.device,
    )
    correction = (effective_weight * s).sum(dim=-1)
    apply_additive_effective_bias_shift(
        perceptron, -correction, baked_flag="_neg_shift_baked",
    )


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


def apply_ttfs_quantized_bias_shift(model, target_tq: int) -> None:
    """Backward-compatible alias for :func:`apply_ttfs_quantization_bias_compensation`."""
    apply_ttfs_quantization_bias_compensation(model, target_tq)


LIF_HALF_STEP_FLAG = "_lif_half_step_baked_into_bias"
SYNC_ENTRY_HALF_STEP_FLAG = "_sync_entry_half_step_folded"


def apply_half_step_entry_fold(model, simulation_steps: int, *, baked_flag: str) -> int:
    """Fold the +theta/(2T) deployed-convention half-step as a TRAINABLE entry
    bias (identical bake math as the TTFS compensation; the QAT owns and may
    train the fold): +theta/(2T) per cycle turns the floor rate grid into
    nearest over the window and head-starts every hop's first fire. Encoders
    skipped, idempotent per ``baked_flag``; returns folds applied.
    """
    folded = 0
    for perceptron in model.get_perceptrons():
        if getattr(perceptron, "is_encoding_layer", False):
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


def apply_sync_exact_entry_half_step(model, simulation_steps: int) -> int:
    """[5v B1] The sync half-step entry fold before the exact-ceil QAT. Idempotent;
    returns folds applied."""
    return apply_half_step_entry_fold(
        model, simulation_steps, baked_flag=SYNC_ENTRY_HALF_STEP_FLAG,
    )


def _is_perceptron(node) -> bool:
    return getattr(node, "perceptron", None) is not None


def _assert_baked_encoder_feeds_no_compute_op(node, consumers, compute_op_type) -> None:
    """A baked *subsumed encoder*'s host-op value path consumes the raw
    (unshifted) input, so a downstream ComputeOp would read an uncompensated
    ``B' = B − W·s`` value — fail loud on that topology."""
    if not getattr(node.perceptron, "is_encoding_layer", False):
        return
    frontier = list(consumers.get(id(node), []))
    while frontier:
        c = frontier.pop()
        if _is_perceptron(c):
            continue
        if isinstance(c, compute_op_type):
            raise NotImplementedError(
                "negative-shift: a shifted boundary feeds a subsumed encoder whose "
                "value output is consumed by a ComputeOp; the encoder's host value "
                "path would be uncompensated. This topology is unsupported."
            )
        frontier.extend(consumers.get(id(c), []))


def _bake_consumer_perceptrons(producer, shift, consumers, compute_op_type) -> bool:
    """Bake the negative-shift bias into each consuming perceptron, aligning the per-channel
    shift through intervening (linear) structural nodes. Fails loud on a ComputeOp consumer."""
    baked = False
    frontier = [(c, shift) for c in consumers.get(id(producer), [])]
    while frontier:
        consumer, sh = frontier.pop()
        if _is_perceptron(consumer):
            _assert_baked_encoder_feeds_no_compute_op(consumer, consumers, compute_op_type)
            apply_negative_shift_bias(consumer.perceptron, sh.reshape(-1))
            baked = True
        elif isinstance(consumer, compute_op_type):
            raise NotImplementedError(
                "negative-shift: a ComputeOp output feeding another ComputeOp is "
                "unsupported (no consuming perceptron bias to compensate the shift)."
            )
        else:
            try:
                aligned = consumer.forward(sh.unsqueeze(0)).squeeze(0)
            except Exception as exc:  # pragma: no cover - fail loud with context
                raise NotImplementedError(
                    f"negative-shift: cannot align shift through "
                    f"{type(consumer).__name__}: {exc}"
                ) from exc
            for c in consumers.get(id(consumer), []):
                frontier.append((c, aligned))
    return baked


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


def calibration_forward_for_mode(spiking_mode: str):
    """NF forward that produces ``spiking_mode``'s boundary values for calibration.

    The shift must live in the same domain the mode's encoder clamps, so each
    mode calibrates through its own NF forward (resolved by the mode policy)."""
    if spiking_mode not in _NEG_SHIFT_SUPPORTED_MODES:
        raise NotImplementedError(
            f"negative_value_shift is not implemented for spiking_mode={spiking_mode!r}"
        )
    from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode

    return policy_for_spiking_mode(spiking_mode).calibration_forward()


def apply_negative_value_shifts(
    model, calibration_x: torch.Tensor, T: int, *, forward_fn=None,
) -> dict:
    """Pre-mapping: calibrate per-ComputeOp minima, derive positive shifts, bake the
    consuming perceptron(s), and tag each shifted ``ComputeOpMapper`` with ``_negative_shift``.
    Returns ``{ComputeOpMapper: shift_np}`` (empty if no boundary goes negative)."""
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    if forward_fn is None:
        forward_fn = calibration_forward_for_mode("lif")

    mapper_repr = model.get_mapper_repr()
    mapper_repr._ensure_exec_graph()
    deps_map = mapper_repr._deps

    recorder: dict = {}
    with torch.no_grad():
        forward_fn(model, calibration_x, T, compute_min_recorder=recorder)

    consumers: dict[int, list] = {}
    for node in mapper_repr._exec_order:
        for dep in deps_map.get(node, []):
            consumers.setdefault(id(dep), []).append(node)

    out: dict = {}
    for compute_op, mins in recorder.items():
        s = torch.clamp(-mins, min=0.0)
        if not bool((s > 0).any()):
            continue
        if _bake_consumer_perceptrons(compute_op, s, consumers, ComputeOpMapper):
            compute_op._negative_shift = s.detach().cpu().numpy()
            out[compute_op] = compute_op._negative_shift
    return out


def transfer_negative_shifts_to_ir(model, ir_graph) -> None:
    """Copy each ``ComputeOpMapper._negative_shift`` onto its matching IR ``ComputeOp`` (by name),
    so the shift travels with the cached/pickled IR graph to any later hybrid build. A per-instance
    op split over its leading dim emits ``{name}_col{i}`` ops, each taking its leading-index shift row."""
    from mimarsinan.mapping.ir import ComputeOp
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    mapper_repr = model.get_mapper_repr()
    mapper_repr._ensure_exec_graph()
    by_name: dict[str, np.ndarray] = {}
    for node in mapper_repr._exec_order:
        s = getattr(node, "_negative_shift", None)
        name = getattr(node, "name", None)
        if isinstance(node, ComputeOpMapper) and s is not None and name is not None:
            by_name[name] = np.asarray(s, dtype=np.float64)
    for node in ir_graph.nodes:
        if not isinstance(node, ComputeOp) or not node.name:
            continue
        if node.name in by_name:
            node._negative_shift = by_name[node.name]  # pyright: ignore[reportAttributeAccessIssue] — dynamic IR side-channel read via getattr
            continue
        base, sep, col = node.name.rpartition("_col")
        if sep and col.isdigit() and base in by_name:
            s = by_name[base]
            if s.ndim >= 2 and int(col) < s.shape[0]:
                node._negative_shift = np.asarray(  # pyright: ignore[reportAttributeAccessIssue] — dynamic IR side-channel read via getattr
                    s[int(col)], dtype=np.float64,
                ).reshape(-1)


def propagate_negative_shifts_to_hybrid(ir_graph, hybrid_mapping) -> dict:
    """Set ``hybrid_mapping.node_output_shifts`` from the IR ComputeOps' ``_negative_shift``
    so HCM applies the same boundary shift (the consuming core's bias is already baked
    pre-mapping). No-op when no ComputeOp is shifted. Returns the installed table."""
    from mimarsinan.mapping.ir import ComputeOp

    table: dict[int, np.ndarray] = {}
    for node in ir_graph.nodes:
        s = getattr(node, "_negative_shift", None)
        if isinstance(node, ComputeOp) and s is not None:
            table[int(node.id)] = np.asarray(s, dtype=np.float64)
    if table:
        hybrid_mapping.node_output_shifts = {**hybrid_mapping.node_output_shifts, **table}
    return table
