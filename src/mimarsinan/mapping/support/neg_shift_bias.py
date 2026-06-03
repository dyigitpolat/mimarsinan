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


def apply_negative_value_shifts(model, calibration_x: torch.Tensor, T: int) -> dict:
    """Pre-mapping: calibrate per-ComputeOp output minima on ``calibration_x``, derive
    positive-domain shifts, bake the consuming perceptron(s), and tag each shifted
    ``ComputeOpMapper`` with ``_negative_shift`` (consumed by NF + propagated to HCM).
    Returns ``{ComputeOpMapper: shift_np}`` (empty if no boundary goes negative)."""
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    mapper_repr = model.get_mapper_repr()
    mapper_repr._ensure_exec_graph()
    deps_map = mapper_repr._deps

    recorder: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(model, calibration_x, T, compute_min_recorder=recorder)

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
    hybrid build — the only object that flows from mapping to the hybrid-build sites."""
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
        if isinstance(node, ComputeOp) and node.name in by_name:
            node._negative_shift = by_name[node.name]


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
