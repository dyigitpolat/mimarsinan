"""Negative-value shift: make negative-producing ComputeOp boundaries lossless.

A ComputeOp ``F`` that emits negatives is shifted by a per-channel
``s = max(0, −min F)`` so the spike encoder (rates clamped to [0, 1]) is lossless —
it sees ``F(x) + s ≥ 0``. The shift is applied at the boundary by BOTH the torch NF
forward (`chip_aligned_segment_forward`, via the ComputeOp's ``_negative_shift``) and
HCM (`node_output_shifts`); the consuming perceptron's bias is pre-corrected **once**
(`apply_negative_shift_bias`: ``B' = B − W·s``) so both inherit it through the normal
mapping. Structurally mirrors `mimarsinan.mapping.support.ttfs_bias`.
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


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
    """Walk structural consumers of ``producer`` to each consuming perceptron, aligning
    the (per-channel) ``shift`` through each structural node's (linear) forward, and bake
    its bias. Returns whether any perceptron was baked. Fails loud on a ComputeOp
    consumer (no perceptron bias to compensate the shift)."""
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
            # Structural node (reshape/permute/ensure-2d/…): linear, so the shift delta
            # transforms by running its forward. Multi-input structural (concat) unsupported.
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


def calibration_forward_for_mode(spiking_mode: str):
    """NF forward that produces ``spiking_mode``'s boundary values for calibration.

    The shift must live in the same domain the mode's encoder clamps, so each
    mode calibrates through its own NF forward."""
    if spiking_mode == "lif":
        from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

        return chip_aligned_segment_forward
    if spiking_mode == "ttfs_cycle_based":
        return _ttfs_segment_calibration_forward
    if spiking_mode in ("ttfs", "ttfs_quantized"):
        return _analytical_segment_calibration_forward
    raise NotImplementedError(
        f"negative_value_shift is not implemented for spiking_mode={spiking_mode!r}"
    )


def apply_negative_value_shifts(
    model, calibration_x: torch.Tensor, T: int, *, forward_fn=None,
) -> dict:
    """Pre-mapping: calibrate per-ComputeOp output minima on ``calibration_x``, derive
    positive-domain shifts, bake the consuming perceptron(s), and tag each shifted
    ``ComputeOpMapper`` with ``_negative_shift`` (consumed by NF + propagated to HCM).
    ``forward_fn`` is the mode's NF forward (default: LIF chip-aligned); see
    :func:`calibration_forward_for_mode`. Returns ``{ComputeOpMapper: shift_np}``
    (empty if no boundary goes negative)."""
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
    """Copy each ``ComputeOpMapper._negative_shift`` onto its matching IR ``ComputeOp``
    (by name), so the shift travels with the (cached/pickled) IR graph to any later
    hybrid build — the only object that flows from mapping to the hybrid-build sites.

    A per-instance op split over its leading dim emits ``{name}_col{i}`` IR ops;
    each column receives its leading-index row of the (leading, channels) shift."""
    from mimarsinan.mapping.ir import ComputeOp
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    mapper_repr = model.get_mapper_repr()
    mapper_repr._ensure_exec_graph()
    by_name: dict[str, np.ndarray] = {}
    for node in mapper_repr._exec_order:
        s = getattr(node, "_negative_shift", None)
        if isinstance(node, ComputeOpMapper) and s is not None:
            by_name[getattr(node, "name", None)] = np.asarray(s, dtype=np.float64)
    for node in ir_graph.nodes:
        if not isinstance(node, ComputeOp) or not node.name:
            continue
        if node.name in by_name:
            node._negative_shift = by_name[node.name]
            continue
        base, sep, col = node.name.rpartition("_col")
        if sep and col.isdigit() and base in by_name:
            s = by_name[base]
            if s.ndim >= 2 and int(col) < s.shape[0]:
                node._negative_shift = np.asarray(
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
