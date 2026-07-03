"""Per-source activation scales for branching architectures (delegated to each mapper's propagate_source_scale)."""

from mimarsinan.mapping.mappers.scale_propagation import walk_out_scales


def compute_per_source_scales(model_repr):
    walk_out_scales(
        model_repr,
        lambda node, deps, out_scales: node.propagate_source_scale(deps, out_scales),
    )
