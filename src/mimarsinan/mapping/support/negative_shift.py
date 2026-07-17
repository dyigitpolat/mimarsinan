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
    """Idempotently bake ``B' = B − W·s`` (``s`` per-axon or scalar) into ``perceptron``."""
    if getattr(perceptron, "_neg_shift_baked", False):
        return
    effective_weight = PerceptronTransformer().get_effective_weight(perceptron)
    s = torch.as_tensor(
        shift, dtype=effective_weight.dtype, device=effective_weight.device,
    )
    correction = (effective_weight * s).sum(dim=-1)
    PerceptronTransformer().apply_effective_bias_transform(
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
    """Bake the negative-shift bias into each consuming perceptron, aligning the per-channel
    shift through intervening (linear) structural nodes. Fails loud on a ComputeOp consumer
    — BEFORE any bake, so an aborted mixed set cannot leave a baked-unstamped model."""
    plan: list = []
    frontier = [(c, shift) for c in consumers.get(id(producer), [])]
    while frontier:
        consumer, sh = frontier.pop()
        if _is_perceptron(consumer):
            _assert_baked_encoder_feeds_no_compute_op(consumer, consumers, compute_op_type)
            plan.append((consumer.perceptron, sh.reshape(-1)))
        elif isinstance(consumer, compute_op_type):
            raise NotImplementedError(
                "negative-shift: a ComputeOp output feeding another ComputeOp is "
                "unsupported (no consuming perceptron bias to compensate the shift; "
                "negative_value_shift=off subsume-forward handles this topology)."
            )
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
    for perceptron, sh in plan:
        apply_negative_shift_bias(perceptron, sh)
    return bool(plan)



def apply_negative_value_shifts(model, minima: dict) -> dict:
    """The ON mechanism: derive positive shifts from the calibrated per-ComputeOp
    ``minima``, bake the consuming perceptron(s) (``B − W·s``), and tag each shifted
    ``ComputeOpMapper`` with ``_negative_shift``. Returns ``{ComputeOpMapper: shift_np}``
    (empty when no boundary goes negative). Calibration belongs to the policy caller."""
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper

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
        # s is stamped in BUFFER units and baked WITHOUT gauge conversion:
        # the effective weight already folds per_input_scales (= kappa_fold),
        # so W_eff.s == W.(kappa*sigma_wire) exactly — an explicit kappa
        # factor double-counts (measured 1.7x on the armed micro-fixture).
        prev = getattr(compute_op, "_negative_shift", None)
        if prev is not None:
            # Drift verifier: a re-calibration of the value-preserved walk
            # must reproduce the stamp (the once-flag skips the re-bake).
            if not torch.allclose(
                torch.as_tensor(prev, dtype=s.dtype), s, atol=1e-4,
            ):
                raise AssertionError(
                    "negative-shift drift: re-calibration produced a shift "
                    "different from the existing stamp (weights changed after "
                    "the bake?)."
                )
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
