"""The Weight Quantization step's always-on row drive route, resolved from the run's flat config.

On the integer value path (a declared readout law on value-domain cores) with the bias on an always-on row of the
weight grid (no on-chip register, no law-resolution bias, not the register route), every core-mapped perceptron's row
carries a code value of its reader's input grid (``mapping.support.bias_drive``), stamped at the step's entry before
any projection. The route and the stamping live here so a replay of the step's projection (the W38 driver's no-update
check) applies the same stamps from the same flat config.

[U-BRANGE2] The drive site reads its row budget from the same declarations the projection reads: one row without
bias-row splitting, else the rows the bias needs (at least a declared count) within the declared cores' axons less one
data axon (``row_budget``); a perceptron that needs more is refused by name.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics, resolve_core_semantics
from mimarsinan.mapping.lanes.model_lanes import lane_absorbed
from mimarsinan.mapping.support.bias_drive import (
    BiasRowDriveStamps,
    bias_rows_take_a_drive,
    clear_bias_row_drives,
    stamp_bias_row_drives,
)
from mimarsinan.mapping.support.drive_rule import RowBudget
from mimarsinan.mapping.support.grid_declarations import projection_grids
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
    resolve_bias_mode,
    resolve_bias_row_splitting,
    resolve_wq_two_scale_projection,
)

__all__ = ["bias_row_drive_route", "row_budget", "stamp_and_report_drives", "stamp_projection_drives"]


def bias_row_drive_route(config: Mapping[str, Any]) -> bool:
    """Whether Weight Quantization stamps always-on row drives for the flat ``config``: its grids resolved as the
    step resolves them (``projection_grids`` with the on-chip bias and the two-scale key), the register route
    (``resolve_wq_two_scale_projection``) and the core semantics."""
    config = dict(config)
    on_chip = resolve_bias_mode(config) == "on_chip"
    grids = projection_grids(config, two_scale=bool(config.get("wq_two_scale_projection", False)),
                             on_chip_bias=lambda: on_chip)
    return bias_rows_take_a_drive(
        config, on_chip_bias=on_chip, law_bias=grids.law_bias,
        register_bias=bool(resolve_wq_two_scale_projection(config)),
        value_domain=is_mvm_core_semantics(resolve_core_semantics(config)))


def row_budget(config: Mapping[str, Any]) -> RowBudget:
    """The drive site's row budget for ``config``: one row where no bias-row splitting is active (the projection keeps
    one row on the shared grid), else the rows the bias needs, at least the declared count under ``fixed``, within
    the largest declared core's axons less one data axon."""
    config = dict(config)
    splitting = resolve_bias_row_splitting(config)
    if not splitting.active:
        return RowBudget(split=False)
    cores = build_platform_constraints_resolved(config).get("cores") or []
    axons = [int(core["max_axons"]) for core in cores if not core.get("has_bias") and core.get("max_axons")]
    floor = int(splitting.fixed_rows) if splitting.mode == "fixed" else 1
    return RowBudget(split=True, floor=max(1, floor), axons=max(axons) if axons else None)


def stamp_projection_drives(model, config: Mapping[str, Any]) -> Optional[BiasRowDriveStamps]:
    """The step's drive stamping on ``model`` for ``config``: on the route, every core-mapped perceptron's drive at its
    reader's grid (the declared lanes' absorbed nodes passed through); elsewhere every stamp removed. None off the
    route."""
    if not bias_row_drive_route(config):
        clear_bias_row_drives(model)
        return None
    # [U-BRANGE2] each row at the code of its reader's grid whose row grid departs least from its bias, on the
    # projection's own weight width, scale rule and row count.
    on_chip = resolve_bias_mode(dict(config)) == "on_chip"
    grids = projection_grids(dict(config), two_scale=bool(config.get("wq_two_scale_projection", False)),
                             on_chip_bias=lambda: on_chip)
    return stamp_bias_row_drives(model, absorbed=lane_absorbed(model, dict(config)),
                                 weight_bits=int(config["weight_bits"]), fractional=grids.fractional_scale,
                                 budget=row_budget(config))


def stamp_and_report_drives(model, config: Mapping[str, Any]) -> Optional[BiasRowDriveStamps]:
    """Weight Quantization's entry call: :func:`stamp_projection_drives`, before any projection, so every projection of
    the run (rungs, probes, recovery) codes the row on one lattice; its stamps printed in the step's log."""
    stamps = stamp_projection_drives(model, config)
    if stamps is None:
        return None
    if stamps.drives:
        drives = stamps.drives.values()
        print(f"[WeightQuantizationStep] always-on bias rows at the drive rule's codes of their reader's input grid "
              f"on {len(stamps.drives)} perceptron(s) (drives {min(drives):.6g} to {max(drives):.6g})")
    if stamps.rows:
        print(f"[WeightQuantizationStep] biases on more than one always-on row: {len(stamps.rows)} perceptron(s), "
              f"{sum(stamps.rows.values()) - len(stamps.rows)} extra row(s), at most {max(stamps.rows.values())}")
    if stamps.undetermined:
        print(f"[WeightQuantizationStep] always-on bias rows kept at 1.0 on {len(stamps.undetermined)} "
              f"perceptron(s) whose reader grid is not determinable here: {list(stamps.undetermined)[:8]}")
    return stamps
