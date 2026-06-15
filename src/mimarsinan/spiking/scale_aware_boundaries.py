"""Scale-aware TTFS boundaries: theta_out normalizes a block's output to [0,1];
the downstream input_scale un-normalizes the [0,1] spike train back to values."""

from __future__ import annotations

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper
from mimarsinan.spiking.segment_partition import perceptron_of


def _as_model_repr(model_repr_or_model):
    """Accept either a ModelRepresentation or a model exposing get_mapper_repr."""
    if hasattr(model_repr_or_model, "_ensure_exec_graph"):
        return model_repr_or_model
    return model_repr_or_model.get_mapper_repr()


def _mean_source_scale(deps, out_scales, default):
    """Mean of the propagated out-scales of a node's sources (default if none)."""
    present = [out_scales[d] for d in deps if d in out_scales]
    if not present:
        return default
    return sum(present) / len(present)


def propagate_boundary_input_scales(model_repr_or_model, input_data_scale: float = 1.0):
    """Forward-propagate theta_out so each perceptron's ``input_activation_scale``
    equals the mean theta_out of its upstream perceptron source(s); the input
    boundary uses ``input_data_scale``. Mirrors the forward walk of
    :func:`compute_per_source_scales` (do not duplicate the graph walk)."""
    model_repr = _as_model_repr(model_repr_or_model)
    model_repr._ensure_exec_graph()

    out_scales: dict = {}

    for node in model_repr._exec_order:
        deps = model_repr._deps.get(node, [])

        if isinstance(node, InputMapper):
            out_scales[node] = float(input_data_scale)

        elif perceptron_of(node) is not None:
            perceptron = perceptron_of(node)
            in_scale = _mean_source_scale(deps, out_scales, float(input_data_scale))
            perceptron.set_input_activation_scale(in_scale)
            out_scales[node] = float(perceptron.activation_scale)

        elif isinstance(node, (ConcatMapper, ComputeOpMapper)):
            out_scales[node] = _mean_source_scale(
                deps, out_scales, float(input_data_scale)
            )

        else:  # transparent routing (reshape/permute/ensure-2d): pass the scale through
            present = [out_scales[d] for d in deps if d in out_scales]
            if present:
                out_scales[node] = sum(present) / len(present)


def calibrate_scale_aware_boundaries(model, activation_scales, input_data_scale: float = 1.0):
    """Set each block's ``activation_scale`` to its distribution-grounded theta_out,
    then propagate so every input un-normalizes from [0,1] (input_scale = upstream
    theta_out). Generic; no model-specific logic."""
    perceptrons = list(model.get_perceptrons())
    if len(activation_scales) != len(perceptrons):
        raise ValueError(
            f"activation_scales count {len(activation_scales)} != perceptron count "
            f"{len(perceptrons)}"
        )
    for perceptron, scale in zip(perceptrons, activation_scales):
        perceptron.set_activation_scale(float(scale))

    propagate_boundary_input_scales(model, input_data_scale=input_data_scale)
