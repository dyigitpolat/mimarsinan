"""Pure snapshot extractors for the GUI monitoring system.

Each function accepts a pipeline artifact and returns a JSON-serializable
summary. Heavy artifacts (PNG heatmaps, per-stage connectivity arrays) are
returned alongside the summary as a list of
:class:`~mimarsinan.gui.resources.ResourceDescriptor` objects so the GUI
server can serve them lazily through dedicated HTTP endpoints.
"""

from mimarsinan.gui.resources import ResourceDescriptor
from mimarsinan.gui.snapshot.builders import (
    RESOURCE_KIND_CONNECTIVITY,
    RESOURCE_KIND_HARD_CORE_HEATMAP,
    RESOURCE_KIND_IR_BANK_HEATMAP,
    RESOURCE_KIND_IR_CORE_HEATMAP,
    RESOURCE_KIND_IR_CORE_PRE_PRUNING,
    RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
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
