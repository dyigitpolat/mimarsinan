"""GUI snapshot module."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mimarsinan.gui")

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




def snapshot_adaptation_manager(manager: Any) -> dict:
    """Extract current adaptation rates."""
    return {
        "clamp_rate": getattr(manager, "clamp_rate", None),
        "shift_rate": getattr(manager, "shift_rate", None),
        "quantization_rate": getattr(manager, "quantization_rate", None),
        "scale_rate": getattr(manager, "scale_rate", None),
        "noise_rate": getattr(manager, "noise_rate", None),
    }

