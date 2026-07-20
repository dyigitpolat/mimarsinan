"""Scale-aware TTFS boundaries: theta_out normalizes a block output to [0,1]; the downstream input_scale un-normalizes it."""

from __future__ import annotations

import torch

from mimarsinan.mapping.mappers.scale_propagation import walk_out_scales
from mimarsinan.spiking.segment_partition import (
    partition_spike_segments,
    perceptron_of,
)


def _as_model_repr(model_repr_or_model):
    """Accept either a ModelRepresentation or a model exposing get_mapper_repr."""
    if hasattr(model_repr_or_model, "_ensure_exec_graph"):
        return model_repr_or_model
    return model_repr_or_model.get_mapper_repr()


def read_boundary_out_scales(model_repr_or_model, input_data_scale: float) -> dict:
    """Pure (no-mutation) twin of :func:`propagate_boundary_input_scales`.

    A perceptron-bearing node yields its activation_scale (mean-collapsed);
    every other node DELEGATES to its own ``propagate_boundary_scale`` — one
    polymorphic implementation for both walks (armed buffer gauges, traffic
    lifts, residual-merge rules), so the two tables cannot drift again
    (calculus §11.2, the one-writer law).
    """
    model_repr = _as_model_repr(model_repr_or_model)
    default = float(input_data_scale)

    def visit(node, deps, out_scales):
        perceptron = getattr(node, "perceptron", None)
        if perceptron is not None:
            scale = perceptron.activation_scale
            if isinstance(scale, torch.Tensor):
                return float(scale.detach().to(torch.float64).mean())
            return float(scale)
        return node.propagate_boundary_scale(deps, out_scales, default)

    return walk_out_scales(model_repr, visit)


def propagate_boundary_input_scales(model_repr_or_model, input_data_scale: float):
    """Forward-propagate theta_out so each perceptron's ``input_activation_scale``
    equals the mean theta_out of its upstream perceptron source(s); the input
    boundary uses ``input_data_scale``.

    The scale is also stamped on the repr (``input_boundary_scale``) so pure
    re-reads (e.g. the LIF segment policy) agree with the propagated values by
    construction — one value, both walks (the NF↔SCM parity contract).
    """
    model_repr = _as_model_repr(model_repr_or_model)
    default = float(input_data_scale)
    walk_out_scales(
        model_repr,
        lambda node, deps, out_scales: node.propagate_boundary_scale(
            deps, out_scales, default
        ),
    )
    model_repr.input_boundary_scale = default


def stamped_input_boundary_scale(model_repr_or_model) -> float:
    """The scale stamped by the last propagation; 1.0 (unit range) before any."""
    return float(
        getattr(_as_model_repr(model_repr_or_model), "input_boundary_scale", 1.0)
    )


_COHERENCE_RTOL = 1e-3


def verify_boundary_currency_coherence(
    model_repr_or_model, *, input_data_scale: float | None = None,
) -> None:
    """Install-seam coherence certificate (calculus §11.2): every host-fed
    segment entry's re-encode currency (the boundary table) must equal the
    consumer's trained entry currency (``input_activation_scale``); fail loud
    with the offending edges."""
    model_repr = _as_model_repr(model_repr_or_model)
    model_repr._ensure_exec_graph()
    default = (
        float(input_data_scale)
        if input_data_scale is not None
        else stamped_input_boundary_scale(model_repr)
    )
    table = read_boundary_out_scales(model_repr, input_data_scale=default)
    exec_order, deps = model_repr._exec_order, model_repr._deps
    seg_of, produces = partition_spike_segments(exec_order, deps)

    mismatches = []
    for node in exec_order:
        p = perceptron_of(node)
        if p is None or not produces.get(node, False):
            continue
        boundary_deps = [
            d for d in deps.get(node, [])
            if not (produces.get(d, False) and seg_of.get(d) == seg_of.get(node))
        ]
        if not boundary_deps:
            continue
        expected = sum(float(table.get(d, default)) for d in boundary_deps) / len(
            boundary_deps
        )
        stamped = float(
            torch.as_tensor(p.input_activation_scale).detach().to(torch.float64).mean()
        )
        if abs(stamped - expected) > _COHERENCE_RTOL * max(abs(expected), 1e-12):
            mismatches.append(
                f"{type(node).__name__}: entry currency {stamped:.6g} != "
                f"boundary table {expected:.6g}"
            )
    for node in exec_order:
        ps = getattr(node, "per_source_scales", None)
        node_deps = deps.get(node, [])
        if ps is None or len(node_deps) != len(ps):
            continue
        for i, d in enumerate(node_deps):
            expected = float(table.get(d, default))
            actual = float(
                torch.as_tensor(ps[i]).detach().to(torch.float64).mean()
            )
            if abs(actual - expected) > _COHERENCE_RTOL * max(abs(expected), 1e-12):
                mismatches.append(
                    f"{type(node).__name__}: per_source[{i}] {actual:.6g} != "
                    f"producer emitted gauge {expected:.6g} [calculus §16.6]"
                )
    if mismatches:
        raise RuntimeError(
            "boundary currency coherence violated (one-writer law, calculus "
            "§11.2) at: " + "; ".join(mismatches)
        )


def calibrate_scale_aware_boundaries(model, activation_scales, input_data_scale: float):
    """Set each block's ``activation_scale`` to its theta_out, then propagate so
    every input un-normalizes from [0,1]. The encoding layer is pinned to
    ``input_data_scale`` (retuning it breaks NF↔SCM deployment parity).
    """
    perceptrons = list(model.get_perceptrons())
    if len(activation_scales) != len(perceptrons):
        raise ValueError(
            f"activation_scales count {len(activation_scales)} != perceptron count "
            f"{len(perceptrons)}"
        )
    for perceptron, scale in zip(perceptrons, activation_scales):
        if getattr(perceptron, "is_encoding_layer", False):
            perceptron.set_activation_scale(float(input_data_scale))
        else:
            perceptron.set_activation_scale(float(scale))

    propagate_boundary_input_scales(model, input_data_scale=input_data_scale)
