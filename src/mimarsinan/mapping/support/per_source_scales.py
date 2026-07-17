"""Per-source activation scales for branching architectures (delegated to each mapper's propagate_source_scale)."""

from mimarsinan.mapping.mappers.scale_propagation import walk_out_scales
from mimarsinan.mapping.support.value_domain import mark_wire_value_ops


def compute_per_source_scales(model_repr):
    # Gauge classification first: the walk's wrap policy consults the marks.
    mark_wire_value_ops(model_repr)
    walk_out_scales(
        model_repr,
        lambda node, deps, out_scales: node.propagate_source_scale(deps, out_scales),
    )
