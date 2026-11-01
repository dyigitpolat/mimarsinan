"""The source walk's per-channel gauge vectors, read without arming or stamping anything (U-SC2).

``compute_per_source_scales`` walks the mapper graph once, arms the wrap slots and stamps each perceptron's fold from the
vector every node emits: one gauge per channel of its output. A twin that must read a wire value at the gauge the
deployed core folds it at needs those vectors, but must not re-arm or re-stamp the graph it walks, so this is the pure
twin of that walk (as ``read_boundary_out_scales`` is the boundary walk's): a perceptron emits its theta per output
channel, an armed host op its stamped emitted gauge, an unarmed host op its source's gauge over its own channels, and
every layout node and the input root their own (pure) rule.
"""

from __future__ import annotations

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.scale_propagation import (
    first_source_scale,
    perceptron_source_out_scale,
    walk_out_scales,
)
from mimarsinan.mapping.support.gauge_channels import align_gauge, gauge_channels

__all__ = ["read_source_gauges"]


def _visit(node, deps, out_scales):
    perceptron = getattr(node, "perceptron", None)
    if perceptron is not None and hasattr(perceptron, "output_channels"):
        return perceptron_source_out_scale(perceptron)
    if isinstance(node, ComputeOpMapper):
        vector = node.output_scale if node.output_scale is not None else first_source_scale(deps, out_scales)
        return None if vector is None else align_gauge(vector.reshape(-1), gauge_channels(node))
    return node.propagate_source_scale(deps, out_scales)


def read_source_gauges(model_repr) -> dict:
    """``node -> its per-channel gauge vector`` at the stamped state of the graph (the vectors the last source walk
    emitted, when nothing changed since)."""
    return walk_out_scales(model_repr, _visit)
