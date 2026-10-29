"""The core-type door rules both doors call over the layout they judge, beside the platform bias rule: why a
declared core type refuses before any training spend (moved from ``entries_write``, which re-exports them)."""

from __future__ import annotations

from typing import Any, List, Mapping

from mimarsinan.mapping.lanes.declaration import lane_operand_conflict
from mimarsinan.mapping.platform.platform_constraints import (
    PER_AXON, PER_SYNAPSE, SIGN_GRANULARITY_FIELD, SignGranularityError, operand_sign_conflict, pool_sign_granularity,
)
from mimarsinan.mapping.platform.weight_cells import weight_cell_errors


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


def platform_residency_errors(config: Mapping[str, Any]) -> List[str]:
    """Why a declared residency plan refuses on the chip's core types, else [] (both doors, beside the lane rule): [G2]
    on a chip that packs per axon the plan plane does not read the sign-split rows the builder places, so the plan's
    period census and the built program's record disagree (outputs exact); refused until the two planes agree. Whether
    the chip packs per axon is ``pool_sign_granularity``'s fold, the one the builder runs; a value the fold refuses is
    refused here in the fold's words."""
    if config.get("residency_plan") is None or not config.get("allow_scheduling"):
        return []
    cores = declared if isinstance(declared := config.get("cores"), (list, tuple)) else ()
    try:
        packs_per_axon = pool_sign_granularity([c for c in cores if isinstance(c, Mapping)]) == PER_AXON
    except SignGranularityError as refusal:
        return [str(refusal)]
    if not packs_per_axon:
        return []
    axon = [i for i, c in enumerate(cores) if isinstance(c, Mapping) and c.get(SIGN_GRANULARITY_FIELD) == PER_AXON]
    return [f"residency_plan is not composed with per-axon weight sign on this tree: cores {axon} declare "
            f"{SIGN_GRANULARITY_FIELD} '{PER_AXON}', so the builder places each mixed-sign row as a W+ and a W- row, "
            "which the residency planner's plan plane does not read: the plan's period census (resident cores, "
            "backing-store cells) and the built program's record disagree, refused until the two planes agree "
            f"(declare '{PER_SYNAPSE}' on every core type, or no residency_plan)"]


def platform_weight_cell_errors(config: Mapping[str, Any]) -> List[str]:
    """[U-C4] Why a core type's weight cells refuse at both doors, else []: a ``cells_per_weight`` above 1 without its
    ``weight_cell_rows``, or rows that do not divide it (``weight_cells.weight_cell_errors``)."""
    return weight_cell_errors(config.get("cores"))
