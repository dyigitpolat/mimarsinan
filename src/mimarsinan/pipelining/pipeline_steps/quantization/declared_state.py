"""The declared quantizer state at Boundary Quantization and Readout Law (WI-5a, WI-5d): one hook per step.

[U-BRANGE2 (d)] Every grid site the toolchain calibrates has a key in the map (an entry, a perceptron's readout, its
consumer groups by their readers, a lane, an operand bank's entry and readout), and the map carries the checkpoint's
trained-grid inventory (``trained``), whose unplaced grids are refused by name at load: a grid a checkpoint was trained
under is declared, never recalibrated. Each hook runs after its step's calibration, so a declared site's grid replaces
the calibrated one."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from mimarsinan.mapping.export.output_law import model_output_perceptrons
from mimarsinan.mapping.lanes.declaration import lanes_executed
from mimarsinan.mapping.lanes.declared_lanes import declare_lane_state
from mimarsinan.mapping.support.grid_declarations import identity_output_law
from mimarsinan.mapping.support.quantizer_state import (
    QuantizerStateError,
    declare_boundary_grids,
    declare_readout_ranges,
    load_quantizer_state,
    perceptron_fx_names,
)
from mimarsinan.mapping.support.quantizer_state_banks import (
    declare_bank_entry_grids,
    declare_bank_readout_ranges,
    declare_group_ranges,
)


def declare_entry_state(model, entries: list, quantizers: list, config: Mapping[str, Any]) -> int:
    """After calibration: the declared entry grids (and operand bank entries) replace the calibrated ones (0 when
    undeclared). On a lane chip (WI-5e) the DAC reads unsigned 2^b codes."""
    state = load_quantizer_state(config.get("quantizer_state"))
    if state is None:
        return 0
    unsigned = lanes_executed(config) is not None
    return (declare_boundary_grids(model, entries, quantizers, state, unsigned_dac=unsigned)
            + declare_bank_entry_grids(model, entries, quantizers, state, unsigned_dac=unsigned))
def declare_readout_state(model, perceptrons: list, quantizers: list, platform: Mapping[str, Any],
                          pools: Sequence[Tuple[str, Any]] = ()) -> Dict[str, int]:
    """After calibration: the declared readout ranges (with a lane program's offsets and pool lane ranges, WI-5e),
    then the identity output law's inert quantizers (an output core writes its exact integers, so its trained forward
    requantizes nothing). The step report's counts."""
    report: Dict[str, int] = {}
    state = load_quantizer_state(platform.get("quantizer_state"))
    if state is not None:
        report["declared_ranges"] = declare_readout_ranges(model, perceptrons, quantizers, state)
        if state.bank_readout_ranges:
            report["declared_bank_ranges"] = declare_bank_readout_ranges(model, perceptrons, quantizers, state)
        if state.group_ranges:
            report["declared_groups"] = declare_group_ranges(model, perceptrons, quantizers, state)
        if state.readout_offsets or state.lane_ranges:
            if lanes_executed(platform) is None:
                raise QuantizerStateError("quantizer_state declares readout offsets or lane ranges, which are lane "
                                          "grids: declare onchip_arithmetic with readout_law")
            report.update(declare_lane_state(model, perceptrons, quantizers, pools, state))
    if identity_output_law(platform.get("readout_law")):
        outputs = model_output_perceptrons(model.get_mapper_repr())
        names = {id(p): name for name, p in perceptron_fx_names(model.get_mapper_repr()).items()}
        declared = sorted(names[id(p)] for p in outputs if state is not None and names[id(p)] in state.readout_ranges)
        if declared:
            raise QuantizerStateError(f"quantizer_state declares readout ranges for {declared}, whose identity "
                                      f"output law reads no range")
        by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
        for perceptron in outputs:
            by_perceptron[id(perceptron)].calibrate(0.0)
        report["identity_outputs"] = len(outputs)
    return report
