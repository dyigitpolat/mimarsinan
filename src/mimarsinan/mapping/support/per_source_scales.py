"""Per-source activation scales for branching architectures.

Sets ``per_input_scales`` on each Perceptron so weight quantization absorbs
heterogeneous channel scales (typically from ``ConcatMapper``).  For
``ComputeOpMapper`` whose inputs carry heterogeneous scales — either across
multiple sources or within one source's channels — populates
``per_source_scales`` / ``output_scale`` slots so emission stamps
:class:`ScaleNormalizingWrapper` around the wrapped module.

The per-node scale decision is the mapper's own ``propagate_source_scale``
(V6 polymorphism): the kind-specific branches live on each ``Mapper`` subclass,
not in an ``isinstance`` chain here.
"""

from mimarsinan.mapping.mappers.scale_propagation import walk_out_scales


def compute_per_source_scales(model_repr):
    walk_out_scales(
        model_repr,
        lambda node, deps, out_scales: node.propagate_source_scale(deps, out_scales),
    )
