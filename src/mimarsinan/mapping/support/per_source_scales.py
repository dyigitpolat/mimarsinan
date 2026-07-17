"""Per-source activation scales for branching architectures (delegated to each mapper's propagate_source_scale)."""

from mimarsinan.mapping.mappers.scale_propagation import walk_out_scales
from mimarsinan.mapping.support.value_domain import (
    clear_wire_value_ops,
    mark_wire_value_ops,
)


def compute_per_source_scales(model_repr, *, arm_wire_value_ops: bool = True):
    """Gauge classification first: the walk's wrap policy consults the marks.

    ``arm_wire_value_ops=False`` (the TTFS-family wires) clears the marks —
    those paths already value-transcode per-op via ``apply_ttfs``; arming
    them would superimpose a second transcode (measured: sync parity 0.0 on
    the torch-mixer exposure cell)."""
    if arm_wire_value_ops:
        mark_wire_value_ops(model_repr)
    else:
        clear_wire_value_ops(model_repr)
    walk_out_scales(
        model_repr,
        lambda node, deps, out_scales: node.propagate_source_scale(deps, out_scales),
    )
