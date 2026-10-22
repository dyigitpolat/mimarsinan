"""E282.S: the per-column law solver, its refusals by name, its grids and its stable digests."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.export.law_grids import CoreGrid, ReadoutLawGridError
from mimarsinan.mapping.export.law_solver import (
    ReadoutLawDeclarationError,
    ReadoutLawPsumError,
    ReadoutLawRatioError,
    ReadoutLawSpec,
    ReadoutLawStaleRangeError,
    install_readout_laws,
    multiplier_and_shift,
    solve_core_law,
)
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.integer_law import (
    IntegerLawRangeError,
    apply_law_int64,
    round_half_away,
)
from unit.chip_simulation.value_run.integer_programs import (
    LAW,
    build_integer_program,
    law_digests_by_core,
    mlp,
)

SPEC = ReadoutLawSpec(adc_bits=40, mult_bits=16, activation_bits=8, weight_bits=8)


def _law_columns(law):
    return dict(m0=law.m0, bias=law.bias, shift=law.shift, lo=law.lo, hi=law.hi,
                adc_shift=law.adc_shift, adc_levels=law.adc_levels)


def _bounds(columns, fan_in, weight_bits=8, input_levels=127):
    """The worst-case accumulator bound of ``fan_in`` rows at the q_min code ``-2^(b-1)``."""
    return [fan_in * (1 << (weight_bits - 1)) * input_levels] * columns


def _clamp_free(law, acc):
    """The law with f_adc's step but no clamp: what an ADC of unbounded levels reads."""
    unclamped = dict(_law_columns(law), adc_levels=np.full(law.columns, (1 << 62) - 1))
    return apply_law_int64(acc, **unclamped)


class TestMultiplierAndShift:
    @pytest.mark.parametrize("ratio", [0.0123, 0.5, 1.0, 3.7, 1e-6])
    def test_the_largest_shift_whose_multiplier_fits(self, ratio):
        m0, shift = multiplier_and_shift(ratio, 16, core="c")
        assert 1 <= m0 <= (1 << 16) - 1
        assert int(round_half_away(torch.tensor(ratio * 2.0 ** (shift + 1), dtype=torch.float64))) > (1 << 16) - 1

    def test_a_ratio_outside_mult_bits_refuses_naming_the_core(self):
        with pytest.raises(ReadoutLawRatioError, match="core 'conv_7'.*mult_bits=4"):
            multiplier_and_shift(40.0, 4, core="conv_7")

    def test_a_non_positive_ratio_refuses(self):
        with pytest.raises(ReadoutLawRatioError):
            multiplier_and_shift(0.0, 16, core="c")


class TestSolveCoreLaw:
    def test_a_dyadic_ratio_reproduces_the_real_valued_definition(self):
        grid = CoreGrid(input_step=1 / 8, input_levels=127, output_step=1 / 32, output_range=127 / 32)
        bias = np.array([12.0, -20.0, 0.0, 7.0])
        law = solve_core_law(core="c", theta=4.0, bias=bias, acc_bounds=_bounds(4, 10), grid=grid, spec=SPEC)
        acc = np.random.default_rng(0).integers(-10 * 127 * 127, 10 * 127 * 127, size=(500, 4), dtype=np.int64)
        got = apply_law_int64(acc, **_law_columns(law))
        real = (grid.input_step * acc + bias) / (4.0 * grid.output_step)
        want = torch.clamp(round_half_away(torch.as_tensor(real)), -127, 127).numpy()
        assert np.array_equal(got, want)

    def test_adc_at_or_above_the_accumulator_width_is_inert(self):
        grid = CoreGrid(input_step=0.01, input_levels=127, output_step=0.02, output_range=2.54)
        wide = solve_core_law(core="c", theta=100.0, bias=np.zeros(3), acc_bounds=_bounds(3, 64), grid=grid,
                              spec=SPEC)
        assert int(wide.adc_shift.max()) == 0
        narrow = solve_core_law(core="c", theta=100.0, bias=np.zeros(3), acc_bounds=_bounds(3, 64), grid=grid,
                                spec=ReadoutLawSpec(adc_bits=6, mult_bits=16, activation_bits=8, weight_bits=8))
        assert int(narrow.adc_shift.min()) > 0
        acc = np.arange(-20000, 20000, 37, dtype=np.int64)[:, None].repeat(3, 1)
        assert not np.array_equal(apply_law_int64(acc, **_law_columns(wide)),
                                  apply_law_int64(acc, **_law_columns(narrow)))

    def test_every_adc_at_or_above_the_width_is_one_identity_law(self):
        grid = CoreGrid(input_step=0.01, input_levels=127, output_step=0.02, output_range=2.54)

        def law(adc_bits):
            spec = ReadoutLawSpec(adc_bits=adc_bits, mult_bits=16, activation_bits=8, weight_bits=8)
            return solve_core_law(core="c", theta=100.0, bias=np.arange(3.0), acc_bounds=[64 * 127 * 127] * 3,
                                  grid=grid, spec=spec)

        assert law(21).digest() == law(40).digest() == law(200).digest() != law(20).digest()

    def test_a_narrow_adc_covers_every_unsaturated_accumulator_of_a_biased_column(self):
        grid = CoreGrid(input_step=0.01, input_levels=127, output_step=0.02, output_range=2.54)
        bias = np.array([203.0, -203.0, 0.0, 120.0])  # b / theta = +-0.8 R: the in-range accumulators shift by b
        spec = ReadoutLawSpec(adc_bits=6, mult_bits=16, activation_bits=8, weight_bits=8)
        law = solve_core_law(core="c", theta=100.0, bias=bias, acc_bounds=[63500] * 4, grid=grid, spec=spec)
        acc = np.arange(-63500, 63501, dtype=np.int64)[:, None].repeat(4, 1)
        assert np.array_equal(apply_law_int64(acc, **_law_columns(law)), _clamp_free(law, acc))
        unsaturated = np.abs(apply_law_int64(acc, **dict(_law_columns(law), adc_levels=None))) < 127
        assert unsaturated[:, 0].sum() > 0 and int(law.adc_shift.max()) > 0

    @pytest.mark.parametrize("seed", range(6))
    def test_the_adc_clamp_never_changes_an_output(self, seed):
        g = np.random.default_rng(seed)
        bound = int(g.integers(500, 40000))
        grid = CoreGrid(input_step=float(g.uniform(0.002, 0.05)), input_levels=127,
                        output_step=float(g.uniform(0.005, 0.05)), output_range=0.0)
        grid = CoreGrid(input_step=grid.input_step, input_levels=127, output_step=grid.output_step,
                        output_range=grid.output_step * 127)
        theta = float(g.uniform(1.0, 200.0))
        bias = np.round(g.normal(0.0, 2.0, size=5) * theta * grid.output_step * 127 / 2)
        spec = ReadoutLawSpec(adc_bits=int(g.integers(3, 12)), mult_bits=16, activation_bits=8, weight_bits=8)
        law = solve_core_law(core="c", theta=theta, bias=bias, acc_bounds=[bound] * 5, grid=grid, spec=spec)
        acc = np.arange(-bound, bound + 1, dtype=np.int64)[:, None].repeat(5, 1)
        assert np.array_equal(apply_law_int64(acc, **_law_columns(law)), _clamp_free(law, acc))

    def test_an_adc_that_holds_the_column_bound_reads_at_step_one(self):
        grid = CoreGrid(input_step=0.01, input_levels=127, output_step=0.02, output_range=2.54)
        spec = ReadoutLawSpec(adc_bits=8, mult_bits=16, activation_bits=8, weight_bits=8)
        law = solve_core_law(core="c", theta=100.0, bias=np.zeros(2), acc_bounds=[127, 1 << 20], grid=grid,
                             spec=spec)
        assert law.adc_shift.tolist()[0] == 0 and law.adc_shift.tolist()[1] > 0

    def test_the_fp64_bound_reads_every_column_bias(self):
        grid = CoreGrid(input_step=1.0, input_levels=127, output_step=1.0, output_range=127.0)
        with pytest.raises(IntegerLawRangeError, match="core 'c'.*2\\^53"):
            solve_core_law(core="c", theta=1.0, bias=np.array([0.0, 2.0 ** 40]), acc_bounds=[127, 127], grid=grid,
                           spec=SPEC)

    def test_a_law_beyond_the_fp64_bound_refuses_naming_the_core(self):
        grid = CoreGrid(input_step=1.0, input_levels=(1 << 15) - 1, output_step=1e-3, output_range=1.0)
        spec = ReadoutLawSpec(adc_bits=64, mult_bits=40, activation_bits=16, weight_bits=16)
        with pytest.raises(IntegerLawRangeError, match="core 'wide'.*2\\^53"):
            solve_core_law(core="wide", theta=1.0, bias=np.zeros(2), acc_bounds=_bounds(2, 4096, 16, (1 << 15) - 1),
                           grid=grid, spec=spec)

    def test_the_law_digest_is_stable_and_input_sensitive(self):
        grid = CoreGrid(input_step=0.01, input_levels=127, output_step=0.02, output_range=2.54)
        def solve(theta):
            return solve_core_law(core="c", theta=theta, bias=np.arange(5.0), acc_bounds=_bounds(5, 64), grid=grid,
                                  spec=SPEC)

        first, again, other = solve(100.0), solve(100.0), solve(101.0)
        assert first.digest() == again.digest() != other.digest()


class TestDigestsOnPrograms:
    def test_the_per_column_digest_is_equal_on_two_runs(self):
        a = build_integer_program(mlp(), (8,), num_classes=4)
        b = build_integer_program(mlp(), (8,), num_classes=4)
        assert law_digests_by_core(a.ir) == law_digests_by_core(b.ir)
        assert law_digests_by_core(a.ir)

    def test_nf_and_ir_laws_agree_over_every_column(self):
        program = build_integer_program(mlp(), (8,), num_classes=4)
        for perceptron in program.model.get_perceptrons():
            stamp = perceptron.scaler.stamped
            assert stamp is not None
        for node in program.ir.nodes:
            if not isinstance(node, NeuralCore):
                continue
            perceptron = program.model.get_perceptrons()[node.perceptron_index]
            stamp = perceptron.scaler.stamped
            start, end = node.perceptron_output_slice
            for name in ("m0", "shift", "bias", "lo", "hi", "output_step", "adc_shift", "adc_levels"):
                assert np.array_equal(stamp.law[name][start:end], getattr(node.readout_law, name)), name


class _ConcatOfCores(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.head = nn.Linear(6, 5), nn.Linear(6, 3), nn.Linear(8, 4)

    def forward(self, x):
        return self.head(torch.cat([self.a(x), self.b(x)], dim=1))


class _ConcatOfHostValues(_ConcatOfCores):
    def forward(self, x):
        return self.head(torch.cat([torch.relu(self.a(x)), torch.relu(self.b(x))], dim=1))


class _ConcatOfCoreAndHost(_ConcatOfCores):
    def forward(self, x):
        return self.head(torch.cat([self.a(x), torch.relu(self.b(x))], dim=1))


class TestGrids:
    def test_producers_joined_by_a_concatenation_share_the_widest_grid(self):
        program = build_integer_program(_ConcatOfCores().eval(), (6,), num_classes=4)
        cores = {n.name: n for n in program.ir.nodes if isinstance(n, NeuralCore)}
        quantizers = [p.scaler for p in program.model.get_perceptrons()]
        steps = {float(c.readout_law.output_step[0]) for c in cores.values() if c.perceptron_index in (0, 1)}
        assert len(steps) == 1
        assert steps.pop() == max(float(quantizers[0].range), float(quantizers[1].range)) / 127

    def test_a_concatenation_of_host_values_has_no_entry_grid_and_refuses(self):
        with pytest.raises(ReadoutLawGridError, match="no armed boundary grid"):
            build_integer_program(_ConcatOfHostValues().eval(), (6,), num_classes=4)

    def test_a_core_mixing_host_and_core_sources_refuses(self):
        with pytest.raises(ReadoutLawGridError, match="both host-sourced and core-sourced"):
            build_integer_program(_ConcatOfCoreAndHost().eval(), (6,), num_classes=4)


class TestInstallRefusals:
    def _program(self):
        return build_integer_program(mlp(), (8,), num_classes=4)

    def test_absent_readout_law_changes_nothing(self):
        program = self._program()
        for node in program.ir.nodes:
            node.__dict__.pop("readout_law", None)
        before = pickle.dumps(program.ir)
        assert install_readout_laws(program.ir, program.model, {}, weight_quantization=True, weight_bits=8) == 0
        assert pickle.dumps(program.ir) == before

    def test_without_activation_bits_refuses(self):
        program = self._program()
        with pytest.raises(ReadoutLawDeclarationError, match="activation_bits"):
            install_readout_laws(program.ir, program.model, {"readout_law": LAW},
                                 weight_quantization=True, weight_bits=8)

    def test_without_weight_quantization_refuses(self):
        program = self._program()
        with pytest.raises(ReadoutLawDeclarationError, match="weight quantization"):
            install_readout_laws(program.ir, program.model, {"readout_law": LAW, "activation_bits": 8},
                                 weight_quantization=False, weight_bits=8)

    def test_a_declared_fixed_range_that_the_quantizers_do_not_carry_refuses_by_name(self):
        law = {"adc_bits": 40, "range": "fixed", "range_value": 0.5, "mult_bits": 16}
        with pytest.raises(ReadoutLawStaleRangeError, match="range_value 0.5"):
            build_integer_program(mlp(), (8,), num_classes=4, readout_law=law)
        program = build_integer_program(mlp(), (8,), num_classes=4, readout_law=law, readout_ranges=[0.5] * 3)
        assert law_digests_by_core(program.ir)

    def test_a_partial_sum_core_refuses(self):
        program = self._program()
        next(n for n in program.ir.nodes if isinstance(n, NeuralCore)).psum_role = "partial_pos"
        with pytest.raises(ReadoutLawPsumError, match="partial-sum core"):
            install_readout_laws(program.ir, program.model, {"readout_law": LAW, "activation_bits": 8},
                                 weight_quantization=True, weight_bits=8)
