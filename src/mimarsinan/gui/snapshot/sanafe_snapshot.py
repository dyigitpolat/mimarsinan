"""GUI snapshot module."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

from mimarsinan.gui.resources import ResourceDescriptor

RESOURCE_KIND_IR_CORE_HEATMAP = "ir_core_heatmap"
RESOURCE_KIND_IR_CORE_PRE_PRUNING = "ir_core_pre_pruning"
RESOURCE_KIND_IR_CORE_BIAS = "ir_core_bias"
RESOURCE_KIND_IR_BANK_HEATMAP = "ir_bank_heatmap"
RESOURCE_KIND_HARD_CORE_HEATMAP = "hard_core_heatmap"
RESOURCE_KIND_CONNECTIVITY = "connectivity"
RESOURCE_KIND_PRUNING_LAYER_HEATMAP = "pruning_layer_heatmap"

LIVENESS_LIVE = "live"
LIVENESS_BIAS_ONLY = "bias_only"
LIVENESS_DEAD_LEGACY = "dead_legacy"


def snapshot_sanafe_simulation(
    report: Any,
) -> tuple[dict, list[ResourceDescriptor]]:
    """Build the snapshot and resource descriptors for the SANA-FE GUI tab."""
    snap = report.to_snapshot_dict() if report is not None else {
        "arch_preset": "", "sample_indices": [], "aggregate": {}, "per_sample": [],
    }
    return snap, []


def _find_ir_graph_promiser(pipeline: Any) -> str | None:
    """Return the step whose ``promises`` includes ``ir_graph`` (owner of the IR resource folder), or None.

    Callers must treat None as "register descriptors locally", not drop them.
    """
    try:
        steps = getattr(pipeline, "steps", ())
    except Exception:
        return None
    for name, s in steps:
        try:
            if "ir_graph" in set(getattr(s, "promises", ())):
                return str(name)
        except Exception:
            continue
    return None

