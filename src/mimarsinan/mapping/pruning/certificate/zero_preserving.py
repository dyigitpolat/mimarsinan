"""Zero-preservation predicate for host ComputeOps: elimination is exact only when act(0) == 0."""

from __future__ import annotations

from typing import Dict, FrozenSet, Set

import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.certificate.errors import (
    CascadeCertificatePreconditionError,
)

# The mvm ComputeOp payloads the converter can emit that satisfy f(0) == 0,
# enumerated from the code: linear/conv mixins pass ReLU/LeakyReLU/GELU
# activation modules, passthrough handles Identity/Dropout, the structural
# mixin emits pooling ops, Flatten is wiring when it survives as an op.
ZERO_PRESERVING_HOST_OP_TYPES: FrozenSet[type] = frozenset({
    nn.Identity,
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.Flatten,
    nn.Dropout,
    nn.Dropout2d,
})

# Named counterexamples: f(0) != 0, a dead column feeding one changes the
# output when eliminated, so its producer columns need an implicit source.
NON_ZERO_PRESERVING_HOST_OP_TYPES: FrozenSet[type] = frozenset({
    nn.Sigmoid,
    nn.Softmax,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm,
})


def _op_label(op: ComputeOp) -> str:
    return f"ComputeOp id={op.id} name={op.name!r} op_type={op.op_type!r}"


def _module_of(op: ComputeOp):
    params = getattr(op, "params", None) or {}
    return params.get("module")


def _numeric_zero_check(op: ComputeOp, module) -> bool:
    """Evaluate the op's own execution seam on an all-zero input; exact test."""
    n_inputs = int(len(op.input_sources.flatten()))
    zeros = torch.zeros(1, n_inputs, dtype=torch.float32)
    with torch.no_grad():
        out = op.execute_on_gathered(zeros)
    return bool((out == 0).all().item())


def is_zero_preserving_host_op(op: ComputeOp) -> bool:
    """True iff this host op provably maps an all-zero input to all zeros.

    Registered zero-preserving types are still CHECKED numerically (a lying
    registry entry raises); unknown op payloads fail loud naming the op.
    """
    module = _module_of(op)
    if module is None:
        if str(getattr(op, "op_type", "")).lower() == "identity":
            return True
        raise CascadeCertificatePreconditionError(
            f"{_op_label(op)} carries no host module and is not a declared "
            "identity relay; zero-preservation is undecidable. Register the "
            "op in ZERO_PRESERVING_HOST_OP_TYPES or NON_ZERO_PRESERVING_HOST_OP_TYPES."
        )
    mod_type = type(module)
    if mod_type in NON_ZERO_PRESERVING_HOST_OP_TYPES:
        return False
    if mod_type in ZERO_PRESERVING_HOST_OP_TYPES:
        if not _numeric_zero_check(op, module):
            raise CascadeCertificatePreconditionError(
                f"{_op_label(op)} module {mod_type.__name__} is registered "
                "zero-preserving but act(0) != 0 on a numeric check; the "
                "registry entry is wrong for this instance."
            )
        return True
    raise CascadeCertificatePreconditionError(
        f"{_op_label(op)} module {mod_type.__name__} is not in the "
        "zero-preservation registry; refusing to guess. Add it to "
        "ZERO_PRESERVING_HOST_OP_TYPES (only if f(0) == 0) or "
        "NON_ZERO_PRESERVING_HOST_OP_TYPES."
    )


def derive_cols_with_implicit_source(
    ir_graph: IRGraph,
) -> Dict[int, FrozenSet[int]]:
    """Producer columns consumed by a non-zero-preserving host op.

    These columns emit act(0) != 0 when their axon inputs die, so the cascade
    must treat them as having an implicit out-of-matrix source (never killed
    by propagation). Unknown ops fail loud via :func:`is_zero_preserving_host_op`.
    """
    neural_ids = {
        n.id for n in ir_graph.nodes if isinstance(n, NeuralCore)
    }
    out: Dict[int, Set[int]] = {}
    for node in ir_graph.nodes:
        if not isinstance(node, ComputeOp):
            continue
        if is_zero_preserving_host_op(node):
            continue
        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id in neural_ids:
                out.setdefault(src.node_id, set()).add(int(src.index))
    return {nid: frozenset(cols) for nid, cols in out.items()}


def assert_zero_preserving_preconditions(ir_graph: IRGraph) -> None:
    """Refuse when any host op is not zero-preserving (or unknown).

    The escape hatch is routing the producer columns into
    ``cols_with_implicit_source`` (see :func:`derive_cols_with_implicit_source`);
    the cascade does not consume that exemption externally yet, so the
    certificate refuses instead of silently passing.
    """
    offenders = []
    for node in ir_graph.nodes:
        if isinstance(node, ComputeOp) and not is_zero_preserving_host_op(node):
            offenders.append(_op_label(node))
    if offenders:
        raise CascadeCertificatePreconditionError(
            "cascade elimination is only exact when every host activation "
            f"maps 0 to 0; non-zero-preserving ops present: {offenders}. "
            "Route their producer columns into cols_with_implicit_source "
            "(derive_cols_with_implicit_source) before eliminating."
        )
