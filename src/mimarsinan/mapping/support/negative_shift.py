"""Negative-boundary shift machinery: residual bakes, stamps, and IR/hybrid transfer."""

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
    """Bake ``B' = B − W·s`` (``s`` per-axon or scalar) into ``perceptron``.

    Delta semantics: callers pass the RESIDUAL shift of the current
    calibration (effective minima make a re-calibration's residual zero, so
    re-entry is naturally idempotent — no once-flag)."""
    effective_weight = PerceptronTransformer().get_effective_weight(perceptron)
    s = torch.as_tensor(
        shift, dtype=effective_weight.dtype, device=effective_weight.device,
    )
    correction = (effective_weight * s).sum(dim=-1)
    PerceptronTransformer().apply_effective_bias_transform(
        perceptron, lambda b, c=correction: b - c,
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
            # Under the producer-side lift a host consumer reads the LIFTED
            # value in every representation (mapper forward on the NF side,
            # the shifted gather on the deployed side): nothing to bake, and
            # the consumer's own boundary owns its own sigma.
            continue
        elif sh.numel() == 1:
            # A scalar shift is axis-invariant: structural reshapes cannot
            # change it, so it passes through without a forward.
            for c in consumers.get(id(consumer), []):
                frontier.append((c, sh))
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



def apply_negative_value_shifts(
    model, minima: dict, *, minima_units: str = "buffer",
) -> dict:
    """The ON mechanism: derive positive RESIDUAL shifts from the calibrated
    per-ComputeOp effective ``minima``, bake the consuming perceptron(s)
    (``B − W·s_delta``), and ACCUMULATE each producer's ``_negative_shift``
    stamp (buffer units). ``minima_units="value"`` converts a value-domain
    calibration (the analytical walk) by the armed producer's output scale.
    Returns ``{ComputeOpMapper: total_shift_np}`` for ops stamped THIS call."""
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

    if minima_units not in ("buffer", "value"):
        raise ValueError(f"minima_units must be buffer|value, got {minima_units!r}")
    if not minima:
        return {}
    mapper_repr = model.get_mapper_repr()
    mapper_repr._ensure_exec_graph()
    deps_map = mapper_repr._deps

    consumers: dict[int, list] = {}
    for node in mapper_repr._exec_order:
        for dep in deps_map.get(node, []):
            consumers.setdefault(id(dep), []).append(node)

    out: dict = {}
    for compute_op, mins in minima.items():
        s = torch.clamp(-mins, min=0.0)
        if not bool((s > 0).any()):
            continue
        s = _to_buffer_units(compute_op, s, minima_units)
        # s is baked WITHOUT gauge conversion: the effective weight already
        # folds per_input_scales (= kappa_fold), so W_eff.s ==
        # W.(kappa*sigma_wire) exactly — an explicit kappa factor
        # double-counts (measured 1.7x on the armed micro-fixture).
        if _bake_consumer_perceptrons(compute_op, s, consumers, ComputeOpMapper):
            prev = getattr(compute_op, "_negative_shift", None)
            total = s.detach().cpu().numpy()
            if prev is not None:
                total = np.asarray(prev, dtype=total.dtype) + total
            compute_op._negative_shift = total
            out[compute_op] = compute_op._negative_shift
    return out


def _to_buffer_units(compute_op, shift: torch.Tensor, units: str) -> torch.Tensor:
    """A value-domain calibration divides by an armed producer's output scale
    (its buffer is wire = value/s_out); plain producers buffer the value."""
    if units == "buffer":
        return shift
    out_scale = getattr(compute_op, "output_scale", None)
    if getattr(compute_op, "per_source_scales", None) is None or out_scale is None:
        return shift
    scale = out_scale.detach().to(dtype=shift.dtype, device=shift.device).reshape(-1)
    if scale.numel() != shift.numel():
        scale = scale.mean()
    return shift / scale.clamp(min=1e-12)


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
            if s.size == 1:
                # A scalar shift is instance-invariant: every split column
                # carries it verbatim.
                node._negative_shift = s.reshape(-1)  # pyright: ignore[reportAttributeAccessIssue] — dynamic IR side-channel read via getattr
            elif s.ndim >= 2 and int(col) < s.shape[0]:
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
