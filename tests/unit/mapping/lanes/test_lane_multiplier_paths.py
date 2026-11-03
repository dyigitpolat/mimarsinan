"""[U-LMP] Every multiplier path of a lane law keeps its path, or the core is refused by name.

The lane law ``y = clamp(rnd_s(M0 f_adc(acc) + M_1 q + B), lo, hi)`` shares one shift ``s`` between the data
multiplier ``M0`` and a fused add's operand multiplier ``M_1`` (the declared lane rounds once). The solver takes the
largest ``s`` at which both fit ``mult_bits``; where the data ratio sits far below the operand's (16-bit weights: a
threshold near 5e4), ``M0`` rounds to 0 and the data path is dropped while every twin reads the same law (W08's
lanes-on program, U-BRG's finding F-1). The rule: a multiplier of 0 is a lost path, and a path the sharing starves must
resolve its reach at the shared shift (``X <= 2^s``: its rounding moves an output by at most half a code over
everything it reaches), else the core is refused by name. Guards: a law that keeps every path, and a law without an
operand, are the base's law integer for integer (their digests pinned from the stacked base).
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.common.refusal import is_toolchain_refusal
from mimarsinan.mapping.export.law_rules import ReadoutLawSpec
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.lanes.ir.fusion import OPERAND_ROWS
from mimarsinan.mapping.lanes.law.lane_grids import CoreLaneGrid, GridRead
from mimarsinan.mapping.lanes.law.solve import solve_lane_law
from unit.chip_simulation.value_run.integer_programs import assert_exact, residual_block_net
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform, real_inputs

REFUSAL = "LawMultiplierPathError"


def _grid(in_step: float, out_step: float, op_step, columns: int = 2) -> CoreLaneGrid:
    op = None if op_step is None else GridRead(float(op_step), 0.0, 15, (0, 15))
    return CoreLaneGrid(inputs=GridRead(float(in_step), 0.0, 15, (0, 15)), output_step=float(out_step),
                        output_offset=0.0, lo=0, hi=15, operands=(op,) * columns)


def _core(theta: float, columns: int = 2, operand: bool = True):
    core = SimpleNamespace(name="add_core", threshold=float(theta), perceptron_index=4)
    if operand:
        setattr(core, OPERAND_ROWS, np.arange(columns) + 64)
    return core


def _solve(theta, in_step, out_step, op_step, *, mult_bits=16, bias=(0.0, 0.0), bound=1 << 24, adc_bits=40):
    columns = len(bias)
    spec = ReadoutLawSpec(adc_bits=adc_bits, mult_bits=mult_bits, activation_bits=4, weight_bits=16)
    return solve_lane_law(core=_core(theta, columns, op_step is not None), bias=np.asarray(bias, dtype=np.float64),
                          acc_bounds=(bound,) * columns, grid=_grid(in_step, out_step, op_step, columns), spec=spec)


def _law_tuple(law) -> tuple:
    return tuple(np.asarray(getattr(law, f.name)).tobytes() for f in fields(law))


def _assert_refused(excinfo, *words):
    assert excinfo.typename == REFUSAL, excinfo.typename
    assert is_toolchain_refusal(excinfo.value)
    text = str(excinfo.value)
    for word in words:
        assert word in text, (word, text)


# -- the defect's ratio: W08's add-lane core (theta 56743, Delta_in 0.2252, Delta_out 0.3530, operand ratio 1.797) ----

W08_THETA, W08_IN, W08_OUT, W08_OPERAND_RATIO = 56743.0, 0.22523508, 0.35297343, 58890 / 32768


def test_a_data_multiplier_rounded_to_zero_is_refused_by_name():
    """W08's add-lane core: at the shift the operand sets (15), ``M0 = rnd(1.12e-5 x 2^15) = 0``: the data path is lost.
    Red before: the law is solved with ``m0 = 0`` and no refusal."""
    with pytest.raises(Exception) as excinfo:
        _solve(W08_THETA, W08_IN, W08_OUT, W08_OPERAND_RATIO * W08_OUT)
    _assert_refused(excinfo, "'add_core'", "perceptron 4", "data multiplier is 0", "lost", "mult_bits 16",
                    "operand", "shift 15")


@pytest.mark.parametrize("adc_bits", [4, 8])
def test_a_data_path_lost_behind_a_narrow_converter_is_refused_by_name(adc_bits):
    """The same core behind a converter narrower than its accumulator: the solve reads each column's reach to the clamp
    (``clamp_reach``) before the law hook, so the lost path must reach the hook there too. Red before (the stack with
    U-LNR 7da906c4): a bare ``ZeroDivisionError`` in ``clamp_reach`` (U-LNR review M-1); U-LNR's targeted pass (3ac3ef4b)
    makes the reach total at ``M0 = 0`` and the hook refuses by name."""
    with pytest.raises(Exception) as excinfo:
        _solve(W08_THETA, W08_IN, W08_OUT, W08_OPERAND_RATIO * W08_OUT, adc_bits=adc_bits)
    _assert_refused(excinfo, "'add_core'", "perceptron 4", "data multiplier is 0", "lost")


def test_a_data_multiplier_below_its_floor_is_refused_and_the_floor_admits_the_next_code():
    """The operand (ratio 1, ``M_1 = 2^15``) sets ``s = 15``; a dyadic data ratio ``m / 2^15`` gives ``M0 = m``. On the
    4-bit unsigned grid with the operand's codes 0..15 the law's reach is about ``475,137 / m`` accumulators: ``m = 15``
    resolves it within ``2^15`` (kept, and equal to the base's law), ``m = 14`` does not (refused by name)."""
    kept = _solve(1.0, 15 / 32768, 1.0, 1.0)
    assert int(np.asarray(kept.m0)[0]) == 15 and int(np.asarray(kept.shift)[0]) == 15
    with pytest.raises(Exception) as excinfo:
        _solve(1.0, 14 / 32768, 1.0, 1.0)
    _assert_refused(excinfo, "data multiplier is 14", "half a code", "floor 2^15")


def test_an_operand_multiplier_rounded_to_zero_is_refused_by_name():
    """A data ratio of 1 sets ``s = 15``; an operand ratio of 2^-17 rounds ``M_1`` to 0: the fused add no longer reads
    its operand. Red before: ``operand_m = 0`` and no refusal."""
    with pytest.raises(Exception) as excinfo:
        _solve(1.0, 1.0, 1.0, 2.0 ** -17)
    _assert_refused(excinfo, "operand (column 0) multiplier is 0", "lost")


def test_an_operand_multiplier_below_its_floor_is_refused_by_name():
    """A data ratio of 2^12 leaves ``s = 3``; the operand (ratio 1, ``M_1 = 8``) reaches code 15 > 2^3, so its rounding
    can move an output by more than half a code."""
    with pytest.raises(Exception) as excinfo:
        _solve(1.0, 4096.0, 1.0, 1.0)
    _assert_refused(excinfo, "operand (column 0) multiplier is 8", "floor 2^3")


# -- guards: what keeps its paths is the base's law, integer for integer ------------------------------------------------

#: The stacked base's laws (sha256 over every LaneLaw field's bytes), read on the pristine R5-candidate copy.
BASE_FIXTURE_LAW_SHA = {
    "route_b_like": "0cea9927cbc456bae78db0437049093ad20dd12e128abf53ed30627a08137854",  # M0 873813, s 23
    "kept_m15": "8c8a0e47edcbfbb9a40e5a0fec48f05c0609edf0d401723432a12b804dffb980",  # M0 15, s 15, M_1 2^15
    "no_operand_w08_ratio": "40e2b3cea84e7d2b200759827aba17f271cf8b68cd6f14926658df91175a6953",  # M0 48299, s 32
}


def _sha(law) -> str:
    h = hashlib.sha256()
    for part in _law_tuple(law):
        h.update(part)
    return h.hexdigest()


@pytest.mark.parametrize("name, args", [
    ("route_b_like", dict(theta=12.0, in_step=0.25, out_step=0.2, op_step=0.31, mult_bits=24, bias=(3.0, -2.0))),
    ("kept_m15", dict(theta=1.0, in_step=15 / 32768, out_step=1.0, op_step=1.0)),
    ("no_operand_w08_ratio", dict(theta=W08_THETA, in_step=W08_IN, out_step=W08_OUT, op_step=None)),
])
def test_a_law_that_keeps_every_path_is_the_base_law(name, args):
    """Guard (green before and after): a law whose paths keep their reach (route (b)'s form: 4-bit weights, mult_bits
    24), the floor's first kept code, and a law without an operand (one path at its own shift, even at W08's data
    ratio) are byte-equal to the stacked base's."""
    law = _solve(args.pop("theta"), args.pop("in_step"), args.pop("out_step"), args.pop("op_step"), **args)
    assert _sha(law) == BASE_FIXTURE_LAW_SHA[name]


# -- programs: the tree's residual-block lane program (lanes L, E, P) --------------------------------------------------

#: The stacked base's residual-block lane program at 4-bit weights and mult_bits 24 (the lane fixtures' default form):
#: sha256 over its laws in IR order.
BASE_PROGRAM_LAW_SHA = "356ad15f113747ff381967886afb6915ef2f514e671b03216b85c1895230a151"


def _program(weight_bits: int, mult_bits: int):
    law = {"adc_bits": 40, "range": "calibrated", "mult_bits": mult_bits}
    return build_lane_program(residual_block_net(), (3, 8, 8), num_classes=10, weight_bits=weight_bits,
                              platform=lane_platform(("L", "E", "P"), activation_bits=4, readout_law=law))


def test_a_lane_program_whose_add_drops_its_data_path_is_refused_at_the_law_hook():
    """16-bit weights at mult_bits 16 (W08's form): both fused adds' data multipliers round to 0 at the operand's shift.
    Red before: the program builds with ``m0 = 0`` on its add cores and no refusal."""
    with pytest.raises(Exception) as excinfo:
        _program(16, 16)
    _assert_refused(excinfo, "data multiplier is 0", "lost")


def test_a_lane_program_that_keeps_every_path_is_unchanged_and_exact():
    """Guard: the lane fixtures' form (4-bit weights, mult_bits 24) keeps every path; its laws are the base's and
    NF = identity = packed = int64 golden at atol 0."""
    program = _program(4, 24)
    h = hashlib.sha256()
    for node in program.ir.nodes:
        if isinstance(node, NeuralCore) and getattr(node, "readout_law", None) is not None:
            for part in _law_tuple(node.readout_law):
                h.update(part)
    assert h.hexdigest() == BASE_PROGRAM_LAW_SHA
    x = real_inputs(program)
    with torch.no_grad():
        scm = program.identity()(x)
        assert_exact(program.nf()(x), scm)
        mapping = program.packed_mapping(axons=256, neurons=32)
        hcm = ValueHybridCoreFlow(mapping, device="cpu", dtype=torch.float64)(x)
    assert_exact(hcm, scm)
    codes, steps = run_golden(mapping, x.numpy())
    assert np.array_equal(codes * steps, hcm.reshape(codes.shape).numpy())


def test_the_refusal_is_a_toolchain_refusal_raised_through_the_law_hook():
    """The refusal is named (a ``ToolchainRefusal``), so a census or a session records it by name."""
    with pytest.raises(Exception) as excinfo:
        _program(16, 16)
    assert is_toolchain_refusal(excinfo.value) and excinfo.typename == REFUSAL
