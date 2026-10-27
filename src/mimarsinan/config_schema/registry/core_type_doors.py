"""The core-type door rules both doors call over the layout they judge, beside the platform bias rule: why a
declared core type refuses before any training spend (moved from ``entries_write``, which re-exports them)."""

from __future__ import annotations

from typing import Any, List, Mapping

from mimarsinan.mapping.lanes.declaration import lane_operand_conflict
from mimarsinan.mapping.platform.platform_constraints import PER_AXON, PER_SYNAPSE, operand_sign_conflict


def platform_sign_errors(config: Mapping[str, Any]) -> List[str]:
    """Why a per-axon core type refuses, else [] (the one rule both doors call over the layout they judge, beside the
    bias rule): under float weights (``pipeline_mode`` 'vanilla' or ``weight_quantization`` false, the derivation's
    reading) its split stores each weight in sign pairs, exact only on the integer grid (``SignSplitError`` names it at
    hard-core mapping, after training); beside an operand core type it is not composed (``operand_sign_conflict``)."""
    cores = declared if isinstance(declared := config.get("cores"), (list, tuple)) else ()
    float_weights = config.get("pipeline_mode") == "vanilla" or config.get("weight_quantization") is False
    named = [f"cores[{i}].weight_sign_granularity='{PER_AXON}' needs weights on the integer grid, and the run "
             "declares float weights (pipeline_mode 'vanilla' or weight_quantization false): a per-axon core stores "
             f"each weight in sign pairs; declare weight quantization or a {PER_SYNAPSE} core type"
             for i, core in enumerate(cores)
             if float_weights and isinstance(core, Mapping) and core.get("weight_sign_granularity") == PER_AXON]
    return named + [c for c in (operand_sign_conflict(cores),) if c is not None]


def platform_lane_errors(config: Mapping[str, Any]) -> List[str]:
    """Why executed lanes refuse beside the chip's core types, else [] (both doors, beside the per-axon rule): an
    operand core type is not composed with lanes on the integer path (``lane_operand_conflict``)."""
    return [c for c in (lane_operand_conflict(config),) if c is not None]

