"""Scale-aware TTFS boundaries: theta_out normalizes a block's output to [0,1];
the downstream input_scale un-normalizes the [0,1] spike train back to values."""

from __future__ import annotations

from mimarsinan.mapping.mappers.scale_propagation import walk_out_scales


def _as_model_repr(model_repr_or_model):
    """Accept either a ModelRepresentation or a model exposing get_mapper_repr."""
    if hasattr(model_repr_or_model, "_ensure_exec_graph"):
        return model_repr_or_model
    return model_repr_or_model.get_mapper_repr()


def propagate_boundary_input_scales(model_repr_or_model, input_data_scale: float = 1.0):
    """Forward-propagate theta_out so each perceptron's ``input_activation_scale``
    equals the mean theta_out of its upstream perceptron source(s); the input
    boundary uses ``input_data_scale``. Shares the polymorphic out-scale walk with
    :func:`compute_per_source_scales` via each mapper's ``propagate_boundary_scale``
    (do not duplicate the graph walk)."""
    model_repr = _as_model_repr(model_repr_or_model)
    default = float(input_data_scale)
    walk_out_scales(
        model_repr,
        lambda node, deps, out_scales: node.propagate_boundary_scale(
            deps, out_scales, default
        ),
    )


def calibrate_scale_aware_boundaries(model, activation_scales, input_data_scale: float = 1.0):
    """Set each block's ``activation_scale`` to its distribution-grounded theta_out,
    then propagate so every input un-normalizes from [0,1] (input_scale = upstream
    theta_out). Generic; no model-specific logic.

    The ENCODING layer is excluded: its output scale is fixed by the hardware
    input spike-encoding contract (the data→spike generation), not free for
    distribution matching to retune. Retuning it to a teacher quantile (e.g. 2.17
    vs the data scale 1.0) silently breaks NF↔SCM deployment parity — the cascade
    decodes the encoded input at the retuned scale but the hardware executor
    encodes at the data scale, diverging worst at the shallow layers that consume
    the encoded input. So the encoding block is pinned to ``input_data_scale``."""
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
