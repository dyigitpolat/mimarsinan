"""[U-LNR] The converter's reach on every lane grid.

Below its accumulator width, a lane law's converter (ADC) step spans the law's reach to its clamp ``[lo, hi]`` on both
sides, over a fused operand's declared codes, as ``solve_core_law`` does on a symmetric grid: every accumulator beyond
that reach reads ``lo`` or ``hi`` whatever the operand, so the converter's own clamp never changes an output and only
its step does (the declared function). Every twin reads the solved law, so NF = SCM = HCM = the int64 golden at atol 0
by construction. Guards (they pin what the rule leaves alone, so they pass before the change too): a symmetric grid's
law without an operand and a law whose converter is at least as wide as its accumulator are the earlier rule's.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

import mimarsinan.mapping.lanes.law.solve as lane_solve
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.export.law_rules import adc_step_exponent
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.lanes.law.lane_grids import data_rows
from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    accumulator_width,
    apply_law_int64,
    grid_levels,
    law_magnitude_bound,
    rnd_shift_int64,
    saturation_reach,
)
from unit.chip_simulation.value_run.integer_programs import assert_exact, residual_block_net
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform, real_inputs

NARROW = {"adc_bits": 4, "range": "calibrated", "mult_bits": 16}


def _mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 10))


def _build(model, shape, classes, law, **extra):
    """The program and every ``solve_lane_law`` call's inputs and law, keyed by core id."""
    calls = {}
    original = lane_solve.solve_lane_law

    def spy(**kwargs):
        law_ = original(**kwargs)
        calls[int(kwargs["core"].id)] = (kwargs, law_)
        return law_

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lane_solve, "solve_lane_law", spy)
        program = build_lane_program(model, shape, num_classes=10,
                                     platform=lane_platform(classes, activation_bits=4, readout_law=law), **extra)
    return program, calls


@pytest.fixture(scope="module")
def clamp_mlp():
    """Two hidden cores on the ReLU lane's clamp grid (codes 0..15) and a symmetric output core, ADC of 4 bits."""
    return _build(_mlp(), (16,), ("L",), NARROW)


@pytest.fixture(scope="module")
def clamp_residual():
    """A residual block on lanes L, E, P at a 4-bit ADC: clamp grids, a fused add's operand, pool lanes."""
    return _build(residual_block_net(), (3, 8, 8), ("L", "E", "P"), NARROW)


# -- reference thresholds by bisection on the int64 law (independent of the solver's closed form) ----------------------

def _out(acc, m0, bias, shift, lo, hi):
    z = np.asarray(acc, dtype=np.int64) * int(m0) + int(bias)
    return np.clip(rnd_shift_int64(z, int(shift)), int(lo), int(hi))


def _first_hi(m0, bias, shift, lo, hi, span=1 << 40) -> int:
    """The smallest acc whose output is ``hi`` (the output is monotone in acc)."""
    a, b = -span, span
    while a < b:
        mid = (a + b) // 2
        if int(_out(mid, m0, bias, shift, lo, hi)) >= hi:
            b = mid
        else:
            a = mid + 1
    return a


def _last_lo(m0, bias, shift, lo, hi, span=1 << 40) -> int:
    """The largest acc whose output is ``lo``."""
    a, b = -span, span
    while a < b:
        mid = (a + b + 1) // 2
        if int(_out(mid, m0, bias, shift, lo, hi)) <= lo:
            a = mid
        else:
            b = mid - 1
    return a


def reference_reach(m0, bias, shift, lo, hi, term=(0, 0)) -> int:
    """Every acc above ``R`` reads ``hi`` and every acc below ``-R`` reads ``lo`` for every operand term in ``term``:
    the term is a bias shift and the output falls as the bias falls, so ``hi`` binds at the smallest term and ``lo``
    at the largest."""
    return max(abs(_first_hi(m0, bias + term[0], shift, lo, hi)), abs(_last_lo(m0, bias + term[1], shift, lo, hi)))


def r4_saturation_reach(m0: int, bias: int, shift: int, hi: int) -> int:
    """R4's ``saturation_reach`` (integer_law.py at 6a5fdfb1), verbatim: the symmetric grid's closed form."""
    half = (1 << (shift - 1)) if shift > 0 else 0
    acc_hi = -((bias - ((hi << shift) - half)) // m0)
    acc_lo = (-(hi << shift) + half - bias) // m0
    return max(abs(acc_hi), abs(acc_lo))


def _old_rule(m0, bias, shift, hi, bound, levels) -> int:
    """R4's narrow-converter exponent of a symmetric grid (``solve_core_law``'s rule)."""
    return adc_step_exponent(min(int(bound), r4_saturation_reach(int(m0), int(bias), int(shift), int(hi))), levels,
                             core="ref")


def _clamp_grid(grid) -> bool:
    return grid.lo != -grid.hi


def _operand_interval(program, core, law, j):
    """The fused operand term's interval ``[M_1 lo_p, M_1 hi_p]`` of column ``j``: its producer law's clamp."""
    row = int(np.asarray(law.operand_row)[j])
    if row < 0:
        return (0, 0)
    source = core.input_sources.flatten()[row]
    producer = {int(n.id): n for n in program.ir.nodes}[int(source.node_id)].readout_law
    m1 = int(np.asarray(law.operand_m)[j])
    ends = (m1 * int(producer.lo[source.index]), m1 * int(producer.hi[source.index]))
    return (min(ends), max(ends))


def _certify(program, x, **geometry):
    with torch.no_grad():
        scm = program.identity()(x)
        assert_exact(program.nf()(x), scm)
        mapping = program.packed_mapping(**geometry)
        hcm = ValueHybridCoreFlow(mapping, dtype=torch.float64)(x)
    assert_exact(hcm, scm)
    codes, steps = run_golden(mapping, x.numpy())
    assert np.array_equal(codes * steps, hcm.reshape(codes.shape).numpy())
    return scm


def _assert_spans_the_clamp_reach(program, calls, *, operands: bool):
    """Every narrow clamp-grid column's exponent is the finest that spans ``min(bound, reach)``, and the rule moves
    some column below the worst-case exponent (the full accumulator); returns the columns checked."""
    checked = finer = 0
    for kwargs, law in calls.values():
        grid, core = kwargs["grid"], kwargs["core"]
        bounds = [int(v) for v in kwargs["acc_bounds"]]
        if not _clamp_grid(grid) or kwargs["spec"].adc_bits >= accumulator_width(max(bounds)):
            continue
        levels = grid_levels(kwargs["spec"].adc_bits)
        for j, bound in enumerate(bounds):
            term = _operand_interval(program, core, law, j)
            if operands != (term != (0, 0)):
                continue
            reach = reference_reach(int(law.m0[j]), int(law.bias[j]), int(law.shift[j]), grid.lo, grid.hi, term)
            want = adc_step_exponent(min(bound, reach), levels, core="ref")
            assert int(law.adc_shift[j]) == want, (core.name, j, int(law.adc_shift[j]), want)
            checked += 1
            finer += want < adc_step_exponent(bound, levels, core="ref")
    assert checked > 0 and finer > 0, (checked, finer)
    return checked


def _assert_converter_clamp_never_binds(program, calls, seed=0):
    """On every narrow column of a clamp grid, the law with the converter's clamp equals the law with an unclamped
    converter of the same step, over the column's accumulators (key points and a sample of ``[-bound, bound]``) and
    its operand's codes."""
    rng = np.random.default_rng(seed)
    seen = set()
    for kwargs, law in calls.values():
        grid, core = kwargs["grid"], kwargs["core"]
        bounds = [int(v) for v in kwargs["acc_bounds"]]
        if not _clamp_grid(grid) or kwargs["spec"].adc_bits >= accumulator_width(max(bounds)):
            continue
        for j, bound in enumerate(bounds):
            term = _operand_interval(program, core, law, j)
            m1 = int(np.asarray(getattr(law, "operand_m", np.zeros(law.columns)))[j])
            key = (int(law.m0[j]), int(law.bias[j]), int(law.shift[j]), int(law.adc_shift[j]), bound, term, m1)
            if key in seen:
                continue
            seen.add(key)
            m0, b, s, t = key[:4]
            levels, step = int(law.adc_levels[j]), 1 << t
            reach = reference_reach(m0, b, s, grid.lo, grid.hi, term)
            points = [0, bound, -bound, levels * step, -levels * step, reach, -reach]
            points += [p + d * step for p in points for d in (-2, -1, 1, 2)]
            acc = np.unique(np.clip(np.concatenate([np.asarray(points), rng.integers(-bound, bound + 1, 4096)]),
                                    -bound, bound)).astype(np.int64)
            q = np.arange(term[0], term[1] + 1, max(m1, 1), dtype=np.int64) if term != (0, 0) else np.zeros(1, np.int64)
            a, u = np.meshgrid(acc, q, indexing="ij")
            kw = dict(m0=m0, bias=b + u, shift=s, lo=grid.lo, hi=grid.hi, adc_shift=t)
            clamped = apply_law_int64(a, adc_levels=levels, **kw)
            unclamped = apply_law_int64(a, adc_levels=bound + step, **kw)
            assert np.array_equal(clamped, unclamped), (core.name, j)
    assert seen


class TestClampReachHelper:
    def test_the_clamp_reach_is_the_exact_threshold_of_the_law(self):
        """``clamp_reach`` equals the reach the int64 law gives by bisection, on clamp, offset and symmetric grids,
        with and without a fused operand's term interval."""
        from mimarsinan.models.nn.activations.integer_law import clamp_reach

        rng = np.random.default_rng(7)
        grids = [(0, 15), (0, 255), (-7, 7), (-3, 12), (-127, 127), (2, 9), (-15, 0), (0, 1)]
        for _ in range(600):
            lo, hi = grids[int(rng.integers(len(grids)))]
            shift = int(rng.integers(0, 22))
            m0 = int(rng.integers(1, 1 << 16))
            bias = int(rng.integers(-(1 << (shift + 8)), 1 << (shift + 8)))
            m1 = int(rng.integers(0, 1 << 15))
            term = [(0, 0), (0, 15 * m1), (-7 * m1, 7 * m1), (3 * m1, 9 * m1)][int(rng.integers(4))]
            got = clamp_reach(m0, bias, shift, lo, hi, term)
            assert got == reference_reach(m0, bias, shift, lo, hi, term), (m0, bias, shift, lo, hi, term)

    def test_the_symmetric_reach_is_the_saturation_reach(self):
        """On a symmetric grid without an operand the clamp reach is R4's ``saturation_reach`` (solve_core_law's)
        for every grid of at least one positive code, and ``saturation_reach`` itself is unchanged."""
        from mimarsinan.models.nn.activations.integer_law import clamp_reach

        rng = np.random.default_rng(3)
        for _ in range(2000):
            hi, shift = int(rng.integers(1, 300)), int(rng.integers(0, 24))
            m0, bias = int(rng.integers(1, 1 << 20)), int(rng.integers(-(1 << (shift + 12)), 1 << (shift + 12)))
            want = r4_saturation_reach(m0, bias, shift, hi)
            assert clamp_reach(m0, bias, shift, -hi, hi) == want == saturation_reach(m0, bias, shift, hi)

    def test_a_clamp_law_refits_its_reach_from_the_folded_bias(self):
        """The offset fold's refit reads the clamp grid's reach too: on ``clamp(rnd_0(acc + B + u), 0, 7)`` with
        B = -20 the output is 7 from acc 27 on and 0 up to acc 20, so a 7-level step of 4 spans it (``7 * 4 >= 27``);
        an operand term down to -30 moves the hi side to acc 57 (``7 * 8 < 57 <= 7 * 16``)."""
        from mimarsinan.mapping.export.entry_offsets import refit_folded_law
        from mimarsinan.mapping.ir.core_law import CoreLaw

        one = np.ones(1)
        law = CoreLaw(input_step=one, m0=np.array([1]), shift=np.array([0]), bias=np.array([-20]),
                      lo=np.array([0]), hi=np.array([7]), output_step=one, adc_shift=np.array([0]),
                      adc_levels=np.array([7]))
        refit = refit_folded_law(law, [100], identity=False, reach_hi=7, reach_lo=0, core="c")
        assert refit.adc_shift.tolist() == [2]
        with_term = refit_folded_law(law, [1000], identity=False, reach_hi=7, reach_lo=0, terms=((-30, 0),),
                                     core="c")
        assert with_term.adc_shift.tolist() == [4]
        assert refit_folded_law(law, [100], identity=False, reach_hi=7, core="c").adc_shift.tolist() == [2], \
            "a symmetric clamp of 7 (lo -7): reach max(27, 13)"


class TestClampGridLaw:
    def test_every_narrow_clamp_column_spans_its_clamp_reach(self, clamp_mlp):
        program, calls = clamp_mlp
        assert _assert_spans_the_clamp_reach(program, calls, operands=False) >= 64

    def test_the_converter_clamp_never_changes_an_output(self, clamp_mlp):
        _assert_converter_clamp_never_binds(*clamp_mlp)

    def test_nf_scm_hcm_and_the_golden_agree_at_atol_zero(self, clamp_mlp):
        program, calls = clamp_mlp
        _assert_spans_the_clamp_reach(program, calls, operands=False)
        _certify(program, real_inputs(program, 64), axons=64, neurons=16)

    @pytest.mark.parametrize("program_fixture", ["clamp_mlp", "clamp_residual"])
    def test_the_fp64_bound_holds_and_falls_with_the_step(self, program_fixture, request):
        """``law_magnitude_bound`` plus the operand term stays below 2^53 on every lane law, and on a narrow clamp
        column it is at most the worst-case exponent's (the converter's output is at most ``|acc| + 2^t``), below it
        somewhere."""
        program, calls = request.getfixturevalue(program_fixture)
        lower = 0
        for kwargs, law in calls.values():
            grid = kwargs["grid"]
            bounds = [int(v) for v in kwargs["acc_bounds"]]
            op = max([abs(int(m)) * (0 if g is None else g.levels) for m, g in zip(law.operand_m, grid.operands)],
                     default=0)
            now = law_magnitude_bound(bounds, law.m0, law.bias, law.adc_shift) + op
            assert now < FP64_EXACT_BOUND, kwargs["core"].name
            if not _clamp_grid(grid) or kwargs["spec"].adc_bits >= accumulator_width(max(bounds)):
                continue
            levels = int(law.adc_levels[0])
            worst = [adc_step_exponent(b, levels, core="ref") for b in bounds]
            then = law_magnitude_bound(bounds, law.m0, law.bias, worst) + op
            assert now <= then
            lower += now < then
        assert lower > 0


def _core_codes(program, x: torch.Tensor, laws=None):
    """Every core's output codes on ``x`` in IR order, by the int64 law (``laws`` replaces a core's law)."""
    flat = x.reshape(x.shape[0], -1).numpy()
    out = {}
    for core in [n for n in program.ir.nodes if isinstance(n, NeuralCore)]:
        sources = core.input_sources.flatten()
        rows = np.zeros((flat.shape[0], len(sources)), dtype=np.int64)
        grid = getattr(core, "boundary_grid", None)
        entry = grid.entry_codes(flat) if grid is not None and grid.armed else None
        for r, s in enumerate(sources):
            if s.is_input():
                rows[:, r] = entry[:, s.index]
            elif s.node_id >= 0:
                rows[:, r] = out[int(s.node_id)][:, s.index]
        matrix = np.asarray(core.get_core_matrix(program.ir), dtype=np.float64)
        assert np.array_equal(matrix, np.round(matrix))
        mask = data_rows(core)
        acc = rows[:, mask] @ matrix[mask].astype(np.int64)
        law = (laws or {}).get(int(core.id), core.readout_law)
        operand = np.asarray(getattr(law, "operand_row", np.full(law.columns, -1)), dtype=np.int64)
        ops = (((np.where(operand >= 0, rows[:, np.maximum(operand, 0)], 0)), law.operand_m),) \
            if (operand >= 0).any() else ()
        out[int(core.id)] = apply_law_int64(acc, m0=law.m0, bias=law.bias, shift=law.shift, lo=law.lo, hi=law.hi,
                                            adc_shift=law.adc_shift, adc_levels=law.adc_levels, operands=ops)
    return out


def _output_values(program, codes):
    by_id = {int(n.id): n for n in program.ir.nodes}
    cols = []
    for s in program.ir.output_sources.flatten():
        law = by_id[int(s.node_id)].readout_law
        offset = float(np.asarray(getattr(law, "output_offset", np.zeros(law.columns)))[s.index])
        cols.append(codes[int(s.node_id)][:, s.index] * float(law.output_step[s.index]) - offset)
    return np.stack(cols, axis=1)


class TestZeroFractionAndOutputs:
    def test_the_zero_fraction_and_the_deployed_outputs_change_as_the_reach_rule_predicts(self, clamp_mlp):
        """On 256 inputs: the int64 chain of the solved laws is the deployed program (SCM's outputs), each hidden
        clamp core writes fewer exact zeros than the same law at the worst-case exponent (the full accumulator), and
        the network's outputs differ from that law's."""
        program, calls = clamp_mlp
        x = real_inputs(program, 256, seed=5)
        deployed = _core_codes(program, x)
        with torch.no_grad():
            scm = program.identity()(x).numpy()
        assert np.array_equal(_output_values(program, deployed), scm.reshape(scm.shape[0], -1))
        worst = {}
        for cid, (kwargs, law) in calls.items():
            if _clamp_grid(kwargs["grid"]):
                levels = int(law.adc_levels[0])
                worst[cid] = dataclasses.replace(law, adc_shift=np.asarray(
                    [adc_step_exponent(int(b), levels, core="ref") for b in kwargs["acc_bounds"]], dtype=np.int64))
        assert len(worst) == 2, "two hidden clamp cores"
        before = _core_codes(program, x, worst)
        for cid in worst:
            now, then = float((deployed[cid] == 0).mean()), float((before[cid] == 0).mean())
            assert now < then, (calls[cid][0]["core"].name, now, then)
        assert not np.array_equal(_output_values(program, deployed), _output_values(program, before))


class TestOperandLanes:
    def test_an_operand_column_spans_the_clamp_reach_over_its_operand_codes(self, clamp_residual):
        program, calls = clamp_residual
        _assert_spans_the_clamp_reach(program, calls, operands=True)

    def test_the_converter_clamp_never_changes_an_output_with_an_operand(self, clamp_residual):
        _assert_converter_clamp_never_binds(*clamp_residual)

    def test_the_residual_lane_program_certifies_at_atol_zero(self, clamp_residual):
        program, calls = clamp_residual
        _assert_spans_the_clamp_reach(program, calls, operands=False)
        _assert_spans_the_clamp_reach(program, calls, operands=True)
        _certify(program, real_inputs(program, 32), axons=256, neurons=32)


NARROW_ENTRY = {"scale": 3.0, "offset": 1.4, "rounding": "floor", "codes": [0, 15]}


class TestOffsetFold:
    def test_a_clamp_law_beside_an_offset_entry_refits_its_reach_per_channel(self, tmp_path):
        """A padded conv reads the network input through an offset entry grid and its ReLU runs as a lane (a clamp
        law): the fold moves each position's bias, every column's exponent is the largest over its channel's carriers
        of the clamp reach of their folded biases, and the program certifies at atol 0."""
        from pathlib import Path

        from mimarsinan.mapping.export.law_rules import core_bias_and_data_weights
        from mimarsinan.mapping.export.law_stamp import core_channels
        from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
        from mimarsinan.models.nn.activations.integer_law import accumulator_bounds
        from unit.mapping.lanes.lane_programs import write_state

        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 64, 10))
        platform = lane_platform(("L",), activation_bits=5, readout_law={"adc_bits": 6, "range": "calibrated",
                                                                         "mult_bits": 16})

        def declare(flow, entries, grids, perceptrons, quantizers, lanes):
            fx = {id(p): n for n, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
            platform["quantizer_state"] = write_state(Path(tmp_path) / "state.json",
                                                      {fx[id(entries[0])]: {"entry": NARROW_ENTRY}})

        program = build_lane_program(model, (3, 8, 8), num_classes=10, platform=platform, hardware_bias=True,
                                     state_hook=declare)
        first = [n for n in program.ir.nodes if isinstance(n, NeuralCore) and n.perceptron_index == 0]
        assert len(first) > 1
        want, carriers, finer = {}, {}, 0
        for core in first:
            law = core.readout_law
            assert int(law.lo[0]) == 0 and int(law.hi[0]) == 31, "the conv's lane law is a clamp law"
            _bias, weights = core_bias_and_data_weights(core, program.ir, real_bias=True)
            low, high = core.boundary_grid.code_range
            bounds = accumulator_bounds(weights, max(abs(int(low)), abs(int(high))))
            levels = int(law.adc_levels[0])
            for j, (channel, bound) in enumerate(zip(core_channels(core), bounds)):
                e = adc_step_exponent(min(int(bound), reference_reach(int(law.m0[j]), int(law.bias[j]),
                                                                      int(law.shift[j]), 0, 31)), levels, core="ref")
                want[int(channel)] = max(want.get(int(channel), e), e)
                carriers.setdefault(int(channel), set()).add((core.name, j))
                finer += e < adc_step_exponent(int(bound), levels, core="ref")
        assert finer > 0 and any(len(v) > 1 for v in carriers.values())
        for core in first:
            got = [int(t) for t in core.readout_law.adc_shift]
            assert got == [want[int(c)] for c in core_channels(core)], core.name
        _certify(program, real_inputs(program, 32), axons=1024, neurons=256)


class TestGuards:
    """What the rule leaves alone, against R4's rule recomputed here: these pass before the change too."""

    @pytest.mark.parametrize("program_fixture", ["clamp_mlp", "clamp_residual"])
    def test_a_symmetric_grid_without_an_operand_keeps_the_earlier_exponent(self, program_fixture, request):
        program, calls = request.getfixturevalue(program_fixture)
        checked = 0
        for kwargs, law in calls.values():
            grid = kwargs["grid"]
            bounds = [int(v) for v in kwargs["acc_bounds"]]
            if _clamp_grid(grid) or any(op is not None for op in grid.operands):
                continue
            narrow = kwargs["spec"].adc_bits < accumulator_width(max(bounds))
            levels = grid_levels(kwargs["spec"].adc_bits) if narrow else grid_levels(accumulator_width(max(bounds)))
            want = [_old_rule(law.m0[j], law.bias[j], law.shift[j], grid.hi, bound, levels) if narrow else 0
                    for j, bound in enumerate(bounds)]
            assert np.asarray(law.adc_shift).tolist() == want and int(law.adc_levels[0]) == levels
            checked += 1
        assert checked > 0

    @pytest.mark.parametrize("adc_bits", [32, 40])
    def test_a_converter_at_least_as_wide_as_the_accumulator_is_the_identity(self, adc_bits):
        """Every core of the clamp-lane programs at ``adc_bits`` 32 (route (b)'s width) and 40: step 1 at the
        accumulator's own width, on clamp grids, with operands and on symmetric grids alike."""
        law = dict(NARROW, adc_bits=adc_bits)
        for model, shape, classes in ((_mlp(), (16,), ("L",)), (residual_block_net(), (3, 8, 8), ("L", "E", "P"))):
            _program, calls = _build(model, shape, classes, law)
            for kwargs, solved in calls.values():
                width = accumulator_width(max(int(v) for v in kwargs["acc_bounds"]))
                assert width <= adc_bits
                assert np.asarray(solved.adc_shift).tolist() == [0] * solved.columns
                assert np.asarray(solved.adc_levels).tolist() == [grid_levels(width)] * solved.columns


class TestZeroDataMultiplier:
    """[U-LNR targeted pass, review M-1] ``clamp_reach`` is total over the multipliers the solvers produce. A lane law
    takes one shift for its data multiplier M0 and a fused operand's M1, so a data ratio far below the operand's rounds
    M0 to 0: the law then reads no accumulator, has no reach (0), and every converter step is exact, so the solve goes
    on and the rule that refuses a lost path names it. A negative M0 (no solver produces one) is refused by name.
    Before, the reach divided by M0 and a narrow converter stopped with a bare ZeroDivisionError."""

    def test_a_zero_multiplier_has_no_reach_and_its_law_reads_no_accumulator(self):
        from mimarsinan.models.nn.activations.integer_law import clamp_reach

        rng = np.random.default_rng(11)
        acc = np.arange(-(1 << 16), (1 << 16) + 1, 97, dtype=np.int64)
        for _ in range(200):
            lo, hi = [(0, 15), (-7, 7), (-3, 12), (0, 1)][int(rng.integers(4))]
            shift = int(rng.integers(0, 22))
            bias = int(rng.integers(-(1 << (shift + 8)), 1 << (shift + 8)))
            m1 = int(rng.integers(0, 1 << 15))
            term = [(0, 0), (0, 15 * m1), (-7 * m1, 7 * m1)][int(rng.integers(3))]
            assert clamp_reach(0, bias, shift, lo, hi, term) == 0
            for u in term:
                for t, levels in ((0, 7), (6, 7), (14, 1)):
                    y = apply_law_int64(acc, m0=0, bias=bias + u, shift=shift, lo=lo, hi=hi, adc_shift=t,
                                        adc_levels=levels)
                    assert np.all(y == y[0]), "at M0 = 0 the law's output does not depend on the accumulator"
        assert saturation_reach(0, 5, 3, 7) == 0

    def test_a_negative_multiplier_is_refused_by_name(self):
        from mimarsinan.common.refusal import ToolchainRefusal
        from mimarsinan.models.nn.activations.integer_law import IntegerLawRangeError, clamp_reach

        for args in ((-1, 0, 4, 0, 15), (-(1 << 15), 100, 10, -7, 7)):
            with pytest.raises(IntegerLawRangeError, match="M0"):
                clamp_reach(*args)
        with pytest.raises(IntegerLawRangeError, match="M0"):
            saturation_reach(-3, 0, 2, 7)
        assert issubclass(IntegerLawRangeError, ToolchainRefusal)

    @pytest.mark.parametrize("adc_bits", [4, 32])
    def test_a_fused_add_whose_data_multiplier_rounds_to_zero_never_stops_unnamed(self, adc_bits):
        """A lane core whose data ratio is 1e-6 of its operand's (mult_bits 16) takes M0 = 0 under the shared shift. At
        a narrow converter (adc_bits 4) and at the identity converter (32, a guard) it solves with a converter whose
        clamp changes no output, or a rule that refuses the lost data path names the core; never a bare
        ZeroDivisionError."""
        import types

        from mimarsinan.common.refusal import ToolchainRefusal
        from mimarsinan.mapping.export.law_rules import ReadoutLawSpec
        from mimarsinan.mapping.lanes.ir.fusion import OPERAND_ROWS
        from mimarsinan.mapping.lanes.law.lane_grids import CoreLaneGrid, GridRead

        bound = 100_000
        grid = CoreLaneGrid(inputs=GridRead(1e-6, 0.0, 15, (0, 15)), output_step=1.0, output_offset=0.0, lo=0, hi=15,
                            operands=(GridRead(1.0, 0.0, 15, (0, 15)),))
        core = types.SimpleNamespace(name="m0_rounds_to_zero", threshold=1.0, **{OPERAND_ROWS: np.array([1])})
        spec = ReadoutLawSpec(adc_bits=adc_bits, mult_bits=16, activation_bits=5, weight_bits=4)
        try:
            law = lane_solve.solve_lane_law(core=core, bias=np.array([0.0]), acc_bounds=[bound], grid=grid, spec=spec)
        except ToolchainRefusal as refusal:  # a later rule that refuses a lost multiplier path by name
            assert "m0_rounds_to_zero" in str(refusal)
            return
        assert int(law.m0[0]) == 0 and int(law.operand_m[0]) > 0
        width = accumulator_width(bound)
        narrow = adc_bits < width
        assert int(law.adc_shift[0]) == 0
        assert int(law.adc_levels[0]) == (grid_levels(adc_bits) if narrow else grid_levels(width))
        acc = np.unique(np.concatenate([np.arange(-bound, bound + 1, 251), [-bound, -1, 0, 1, bound]])).astype(np.int64)
        a, q = np.meshgrid(acc, np.arange(0, 16, dtype=np.int64), indexing="ij")
        kw = dict(m0=int(law.m0[0]), bias=int(law.bias[0]), shift=int(law.shift[0]), lo=0, hi=15,
                  adc_shift=int(law.adc_shift[0]), operands=((q, int(law.operand_m[0])),))
        clamped = apply_law_int64(a, adc_levels=int(law.adc_levels[0]), **kw)
        assert np.array_equal(clamped, apply_law_int64(a, adc_levels=bound + 1, **kw))
        assert law_magnitude_bound([bound], law.m0, law.bias, law.adc_shift) + int(law.operand_m[0]) * 15 \
            < FP64_EXACT_BOUND

    def test_the_refit_of_a_zero_multiplier_law_spans_no_accumulator(self):
        """The offset fold's refit reads the same total reach: a folded clamp law with M0 = 0 (an operand term down
        to -30) refits its converter to step 1, with no ZeroDivisionError."""
        from mimarsinan.mapping.export.entry_offsets import refit_folded_law
        from mimarsinan.mapping.ir.core_law import CoreLaw

        one = np.ones(1)
        law = CoreLaw(input_step=one, m0=np.array([0]), shift=np.array([4]), bias=np.array([-20]), lo=np.array([0]),
                      hi=np.array([7]), output_step=one, adc_shift=np.array([9]), adc_levels=np.array([7]))
        refit = refit_folded_law(law, [100_000], identity=False, reach_hi=7, reach_lo=0, terms=((-30, 0),), core="c")
        assert refit.adc_shift.tolist() == [0]
