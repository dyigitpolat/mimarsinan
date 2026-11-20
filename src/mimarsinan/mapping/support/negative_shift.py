"""Negative-boundary shift machinery: the policy's shifts and stamps (each reader corrected for the field it receives,
``received_shift``), and their IR/hybrid transfer."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.support.received_shift import bake_received_shift, correct_for_received_shift, every_axon_shift

def negative_shifts_from_min(min_by_node: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-node positive shift ``s = max(0, −min F(x))``; drops all-zero nodes."""
    shifts: dict[int, np.ndarray] = {}
    for node_id, mins in min_by_node.items():
        s = np.clip(-np.asarray(mins, dtype=np.float64), a_min=0.0, a_max=None)
        if np.any(s > 0.0):
            shifts[int(node_id)] = s
    return shifts


def apply_negative_shift_bias(perceptron, shift) -> None:
    """[U-RSHIFT, kept for its importers] ``B' = B - W s`` with ``s`` (one value, or one per input) on every axon of
    ``perceptron``, through the one writer (``received_shift``) as one source: a second call writes nothing, and the
    record (``input_shift``) lets every later weight projection re-derive the compensation on the weights it installs.
    The pipeline's bakes carry each reader's received field instead (``bake_received_shift``)."""
    correct_for_received_shift(perceptron, every_axon_shift(perceptron, shift))


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


def _admitted_readers(producer, consumers, compute_op_type) -> bool:
    """Whether a shift of ``producer`` reaches a perceptron that reads it, through structural nodes only (today's
    admission). Refuses by name a ComputeOp consumer, and a subsumed encoder whose host value feeds a ComputeOp, before
    any bake, so an aborted set leaves no model baked and unstamped. The correction is the one writer's."""
    found = False
    frontier = list(consumers.get(id(producer), []))
    while frontier:
        consumer = frontier.pop()
        if _is_perceptron(consumer):
            _assert_baked_encoder_feeds_no_compute_op(consumer, consumers, compute_op_type)
            found = True
        elif isinstance(consumer, compute_op_type):
            raise NotImplementedError(
                "negative-shift: a ComputeOp output feeding another ComputeOp is "
                "unsupported (no consuming perceptron bias to compensate the shift; "
                "negative_value_shift=off subsume-forward handles this topology)."
            )
        else:
            frontier.extend(consumers.get(id(consumer), []))
    return found



def apply_negative_value_shifts(model, minima: dict) -> dict:
    """The ON mechanism: derive positive shifts from the calibrated per-ComputeOp
    ``minima``, correct every reader for the field it receives from all of them at once
    (``bake_received_shift``), and tag each shifted ``ComputeOpMapper`` with ``_negative_shift``. Returns ``{ComputeOpMapper: shift_np}``
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

    planned: dict = {}
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
            # must reproduce the stamp (a reader that compensates it writes nothing).
            if not torch.allclose(
                torch.as_tensor(prev, dtype=s.dtype), s, atol=1e-4,
            ):
                raise AssertionError(
                    "negative-shift drift: re-calibration produced a shift "
                    "different from the existing stamp (weights changed after "
                    "the bake?)."
                )
        if _admitted_readers(compute_op, consumers, ComputeOpMapper):
            planned[compute_op] = s
    # [U-RSHIFT] every source of this call in one field (sources compose); a source is stamped exactly when its
    # readers compensate it.
    if planned:
        bake_received_shift(mapper_repr, planned)
    out: dict = {}
    for compute_op, s in planned.items():
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
