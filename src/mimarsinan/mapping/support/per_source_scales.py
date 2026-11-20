"""Per-source activation scales for branching architectures (delegated to each mapper's propagate_source_scale)."""

from mimarsinan.mapping.map_model_to_ir import LayoutGaugePlacement, mapper_source_table
from mimarsinan.mapping.mappers.scale_propagation import walk_out_scales
from mimarsinan.mapping.support.gauge_placement import sample_shapes
from mimarsinan.mapping.support.input_root_gauge import fold_input_root_gauge
from mimarsinan.mapping.support.value_domain import (
    clear_wire_value_ops,
    mark_wire_value_ops,
)


def compute_per_source_scales(model_repr, *, arm_wire_value_ops: bool = True, input_gauge: float | None = None):
    """Gauge classification first: the walk's wrap policy consults the marks.

    ``arm_wire_value_ops=False`` (the TTFS-family wires) clears the marks —
    those paths already value-transcode per-op via ``apply_ttfs``; arming
    them would superimpose a second transcode (measured: sync parity 0.0 on
    the torch-mixer exposure cell).

    [U-IN] Under the wire currency every core that reads the raw input folds it at the raw input's wire gauge
    (``input_root_gauge``): ``input_gauge`` as the wire-gauge seam passes it, else the one stamped on the graph.

    [U-PFOLD] A convolution whose fold is held per position reads each element of its input in that input's grid,
    the identity program's per-sample source shape of its source (``gauge_placement.sample_shapes``)."""
    if arm_wire_value_ops:
        mark_wire_value_ops(model_repr)
    else:
        clear_wire_value_ops(model_repr)
    with sample_shapes(lambda node: mapper_source_table(model_repr).shapes[id(node)]):
        out_scales = walk_out_scales(
            model_repr,
            lambda node, deps, out_scales: node.propagate_source_scale(deps, out_scales),
            LayoutGaugePlacement(model_repr),  # [U-TGAUGE3] each layout element at its producer's gauge
        )
        if arm_wire_value_ops:
            fold_input_root_gauge(model_repr, out_scales, input_gauge)
