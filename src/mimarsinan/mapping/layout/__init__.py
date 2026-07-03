"""Shape-only layout estimation for architecture search."""

from mimarsinan.mapping.layout.layout_types import (
    LayoutCoreSnapshot,
    LayoutSoftCoreSpec,
    LayoutHardCoreType,
    LayoutHardCoreInstance,
    LayoutPackingResult,
)
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_plan import LayoutPlan, build_layout_plan
from mimarsinan.mapping.layout.softcore_spec_adapter import (
    spec_from_neural_core,
    spec_from_softcore,
)
from mimarsinan.mapping.layout.segmentation import (
    HostSegment,
    NeuralSegment,
    Segment,
    compute_host_side_segment_count,
    compute_node_latencies,
    compute_segment_ids,
    partition_ir_graph,
)
from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView
from mimarsinan.mapping.layout.layout_source_view_ops import (
    concat_source_views,
    node_ids_of,
    stack_source_views,
    total_size,
)

__all__ = [
    "LayoutCoreSnapshot",
    "LayoutSoftCoreSpec",
    "LayoutHardCoreType",
    "LayoutHardCoreInstance",
    "LayoutPackingResult",
    "LayoutIRMapping",
    "pack_layout",
    "LayoutPlan",
    "build_layout_plan",
    "spec_from_neural_core",
    "spec_from_softcore",
    "HostSegment",
    "NeuralSegment",
    "Segment",
    "compute_host_side_segment_count",
    "compute_node_latencies",
    "compute_segment_ids",
    "partition_ir_graph",
    "LayoutSourceView",
    "concat_source_views",
    "node_ids_of",
    "stack_source_views",
    "total_size",
]
