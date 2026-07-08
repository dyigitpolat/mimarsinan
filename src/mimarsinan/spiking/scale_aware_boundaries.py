"""Scale-aware TTFS boundaries: theta_out normalizes a block output to [0,1]; the downstream input_scale un-normalizes it."""

from __future__ import annotations

import torch

from mimarsinan.mapping.mappers.scale_propagation import (
    mean_source_scale,
    walk_out_scales,
)


def _as_model_repr(model_repr_or_model):
    """Accept either a ModelRepresentation or a model exposing get_mapper_repr."""
    if hasattr(model_repr_or_model, "_ensure_exec_graph"):
        return model_repr_or_model
    return model_repr_or_model.get_mapper_repr()


def read_boundary_out_scales(model_repr_or_model, input_data_scale: float) -> dict:
    """Pure (no-mutation) twin of :func:`propagate_boundary_input_scales`.

    Per-node scalar out-scales: a perceptron-bearing node yields its
    activation_scale (mean-collapsed), every other node passes through the mean
    of its sources' scales — the same aggregation the deployed IR fold bakes
    into consumer weights (per_input_scales).
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
        return mean_source_scale(deps, out_scales, default)

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
