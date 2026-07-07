"""Pure snapshot extractors for the GUI monitoring system: pipeline artifact -> JSON-serializable summary (+ lazy resource descriptors for heavy artifacts)."""

from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.util.constants import (
    RESOURCE_KIND_CONNECTIVITY,
    RESOURCE_KIND_HARD_CORE_HEATMAP,
    RESOURCE_KIND_IR_BANK_HEATMAP,
    RESOURCE_KIND_IR_CORE_HEATMAP,
    RESOURCE_KIND_IR_CORE_PRE_PRUNING,
    RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
)
from mimarsinan.gui.snapshot.builders import (
    build_step_snapshot,
    snapshot_adaptation_manager,
    snapshot_hard_core_mapping,
    snapshot_ir_graph,
    snapshot_model,
    snapshot_pruning_layers,
    snapshot_search_result,
)

__all__ = [
    "ResourceDescriptor",
    "RESOURCE_KIND_CONNECTIVITY",
    "RESOURCE_KIND_HARD_CORE_HEATMAP",
    "RESOURCE_KIND_IR_BANK_HEATMAP",
    "RESOURCE_KIND_IR_CORE_HEATMAP",
    "RESOURCE_KIND_IR_CORE_PRE_PRUNING",
    "RESOURCE_KIND_PRUNING_LAYER_HEATMAP",
    "build_step_snapshot",
    "snapshot_adaptation_manager",
    "snapshot_hard_core_mapping",
    "snapshot_ir_graph",
    "snapshot_model",
    "snapshot_pruning_layers",
    "snapshot_search_result",
]
