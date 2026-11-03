"""The Weight Quantization step's always-on row drive route, resolved from the run's flat config.

On the integer value path (a declared readout law on value-domain cores) with the bias on an always-on row of the
weight grid (no on-chip register, no law-resolution bias, not the register route), every core-mapped perceptron's row
carries the top code value of its reader's input grid (``mapping.support.bias_drive``), stamped at the step's entry
before any projection. The route and the stamping live here so a replay of the step's projection (the W38 driver's
no-update check) applies the same stamps from the same flat config.
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
from mimarsinan.mapping.support.grid_declarations import projection_grids
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    resolve_bias_mode,
    resolve_wq_two_scale_projection,
)

__all__ = ["bias_row_drive_route", "stamp_and_report_drives", "stamp_projection_drives"]


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


def stamp_projection_drives(model, config: Mapping[str, Any]) -> Optional[BiasRowDriveStamps]:
    """The step's drive stamping on ``model`` for ``config``: on the route, every core-mapped perceptron's reader grid
    top (the declared lanes' absorbed nodes passed through); elsewhere every stamp removed. None off the route."""
    if not bias_row_drive_route(config):
        clear_bias_row_drives(model)
        return None
    return stamp_bias_row_drives(model, absorbed=lane_absorbed(model, dict(config)))


def stamp_and_report_drives(model, config: Mapping[str, Any]) -> Optional[BiasRowDriveStamps]:
    """Weight Quantization's entry call: :func:`stamp_projection_drives`, before any projection, so every projection of
    the run (rungs, probes, recovery) codes the row on one lattice; its stamps printed in the step's log."""
    stamps = stamp_projection_drives(model, config)
    if stamps is None:
        return None
    if stamps.drives:
        drives = stamps.drives.values()
        print(f"[WeightQuantizationStep] always-on bias rows at their reader's input full scale on "
              f"{len(stamps.drives)} perceptron(s) (drives {min(drives):.6g} to {max(drives):.6g})")
    if stamps.undetermined:
        print(f"[WeightQuantizationStep] always-on bias rows kept at 1.0 on {len(stamps.undetermined)} "
              f"perceptron(s) whose reader grid is not determinable here: {list(stamps.undetermined)[:8]}")
    return stamps
