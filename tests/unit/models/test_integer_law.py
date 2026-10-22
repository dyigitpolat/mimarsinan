"""E282.L: the integer law's identities, its torch and int64 forms, and one rounding rule everywhere."""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
import torch

from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    adc,
    adc_int64,
    accumulator_bounds,
    accumulator_width,
    affine_after_clamp_bounds,
    apply_law,
    apply_law_int64,
    grid_codes,
    instance_reduce,
    law_magnitude_bound,
    OffGridInputError,
    rnd_shift,
    rnd_shift_int64,
    round_half_away,
    saturation_reach,
)
from mimarsinan.models.nn.activations.value_quantizer import ValueGridQuantizer, quantize_to_value_grid

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


def _half_away_definition(z: np.ndarray, s: int) -> np.ndarray:
    """sign(z) floor(|z| / 2^s + 1/2), in exact integers."""
    if s == 0:
        return z
    return np.sign(z) * ((np.abs(z) + (1 << (s - 1))) >> s)


def _half_even(z: np.ndarray, s: int) -> np.ndarray:
    if s == 0:
        return z
    q, r = np.divmod(z, 1 << s)
    half = 1 << (s - 1)
    return q + ((r > half) | ((r == half) & (q % 2 == 1))).astype(np.int64)


def _law(acc, m0, b, s, lo, hi, operands=()):
    return apply_law_int64(acc, m0=m0, bias=b, shift=s, lo=lo, hi=hi, operands=operands)


class TestRoundingIdentities:
    def test_shift_form_equals_definition_exhaustively(self):
        z = np.arange(-(2 ** 16), 2 ** 16 + 1, dtype=np.int64)
        points = mismatches = torch_mismatches = 0
        for s in range(21):
            hw = rnd_shift_int64(z, s)
            mismatches += int((hw != _half_away_definition(z, s)).sum())
            torch_mismatches += int((rnd_shift(torch.as_tensor(z, dtype=torch.float64), s).numpy() != hw).sum())
            points += z.size
        assert points == 2_752_533
        assert mismatches == 0 and torch_mismatches == 0

    def test_shift_form_on_a_million_random_48_bit_values(self):
        g = np.random.default_rng(20261015)
        z = g.integers(-(2 ** 47), 2 ** 47, size=1_000_000, dtype=np.int64)
        s = g.integers(0, 33, size=z.size)
        hw = rnd_shift_int64(z, s)
        reference = np.empty_like(z)
        for value in range(33):
            k = s == value
            reference[k] = _half_away_definition(z[k], value)
        assert int((hw != reference).sum()) == 0
        assert int((rnd_shift(torch.as_tensor(z, dtype=torch.float64), torch.as_tensor(s)).numpy() != hw).sum()) == 0

    def test_half_to_even_differs_only_on_ties(self):
        z = np.arange(-(2 ** 12), 2 ** 12 + 1, dtype=np.int64)
        for s in range(1, 9):
            differ = rnd_shift_int64(z, s) != _half_even(z, s)
            tie = np.mod(z, 1 << s) == (1 << (s - 1))
            assert differ.any() and bool(np.all(tie[differ]))


class TestLawIdentities:
    def test_max_commutes_with_a_monotone_law(self):
        g = np.random.default_rng(1)
        n = 200_000
        acc = g.integers(-(2 ** 20), 2 ** 20, size=(n, 4), dtype=np.int64)
        m0 = g.integers(1, 2 ** 15, size=(n, 1), dtype=np.int64)
        b = g.integers(-(2 ** 30), 2 ** 30, size=(n, 1), dtype=np.int64)
        acc_t = torch.as_tensor(acc, dtype=torch.float64)
        for sign in (1, -1):
            reduced = instance_reduce(acc_t, torch.as_tensor(sign * m0[:, 0]), mode="max", dim=1).numpy()
            mismatches = (_law(acc, sign * m0, b, 18, -127, 127).max(1)
                          != _law(reduced.astype(np.int64)[:, None], sign * m0, b, 18, -127, 127)[:, 0])
            assert int(mismatches.sum()) == 0
        naive = (_law(acc, -m0, b, 18, -127, 127).max(1)
                 != _law(acc.max(1, keepdims=True), -m0, b, 18, -127, 127)[:, 0])
        assert int(naive.sum()) > 0

    def test_sum_reduction_into_one_accumulator(self):
        g = np.random.default_rng(2)
        q = g.integers(-7, 8, size=(512, 16, 64), dtype=np.int64)
        w = g.integers(-8, 8, size=(64, 100), dtype=np.int64)
        per_instance = torch.as_tensor(np.einsum("bnc,co->bno", q, w), dtype=torch.float64)
        lhs = instance_reduce(per_instance, 1, mode="sum", dim=1).numpy()
        rhs = np.einsum("bc,co->bo", q.sum(1), w)
        assert int((lhs != rhs).sum()) == 0

    def test_operand_on_the_output_grid_passes_the_rounding(self):
        g = np.random.default_rng(3)
        bad = 0
        for s in (1, 7, 15, 24):
            a = g.integers(-(2 ** 40), 2 ** 40, size=500_000, dtype=np.int64)
            q = g.integers(-127, 128, size=a.size, dtype=np.int64)
            bad += int((rnd_shift_int64(a + q * (1 << s), s) != rnd_shift_int64(a, s) + q).sum())
            through_law = _law(a, 1, 0, s, -(2 ** 62), 2 ** 62, operands=((q, 1 << s),))
            bad += int((through_law != rnd_shift_int64(a, s) + q).sum())
        assert bad == 0

    def test_batchnorm_after_relu_and_relu6_as_signed_clamps(self):
        g = torch.Generator().manual_seed(4)
        z = torch.randn(4096, 64, generator=g, dtype=torch.float64) * 3.0
        u = torch.randn(64, generator=g, dtype=torch.float64)
        c = torch.randn(64, generator=g, dtype=torch.float64)
        for hi in (float("inf"), 6.0):
            lo_b, hi_b = affine_after_clamp_bounds(u, c, 0.0, hi)
            reference = u * torch.clamp(z, 0.0, hi) + c
            folded = torch.minimum(torch.maximum(u * z + c, lo_b), hi_b)
            assert float((reference - folded).abs().max()) == 0.0


class TestTorchAndInt64Forms:
    def test_the_two_forms_agree_at_atol_zero(self):
        g = np.random.default_rng(5)
        acc = g.integers(-(2 ** 30), 2 ** 30, size=(256, 32), dtype=np.int64)
        law = dict(m0=g.integers(1, 2 ** 16, 32), bias=g.integers(-(2 ** 40), 2 ** 40, 32),
                   shift=g.integers(0, 40, 32), lo=-127, hi=127,
                   adc_shift=g.integers(0, 6, 32), adc_levels=(1 << 15) - 1)
        torch_form = apply_law(torch.as_tensor(acc, dtype=torch.float64), **law).numpy()
        assert np.array_equal(torch_form.astype(np.int64), apply_law_int64(acc, **law))

    def test_adc_at_or_above_the_accumulator_width_is_the_identity(self):
        bound = 150 * 128 * 127  # q_min = -2^(b-1) is the largest weight-code magnitude
        width = accumulator_width(bound)
        acc = np.random.default_rng(6).integers(-bound, bound + 1, size=10_000, dtype=np.int64)
        levels = (1 << (width - 1)) - 1
        assert np.array_equal(adc_int64(acc, 0, levels), acc)
        assert torch.equal(adc(torch.as_tensor(acc, dtype=torch.float64), 0, levels),
                           torch.as_tensor(acc, dtype=torch.float64))
        narrow = adc_int64(acc, 4, (1 << 7) - 1)
        assert not np.array_equal(narrow, acc) and np.all(narrow % 16 == 0)

    def test_the_accumulator_bound_is_exact_per_column_and_counts_q_min(self):
        weights = np.array([[-128.0, 3.0, 0.0], [127.0, -1.0, 0.0], [-128.0, 0.0, 0.0]])
        assert accumulator_bounds(weights, 127) == [383 * 127, 4 * 127, 0]
        codes = np.array([[-127, 127, 127], [127, -127, -127], [-127, 127, 127]], dtype=np.int64)
        assert int(np.abs(codes.T @ weights.astype(np.int64)).max()) == 383 * 127
        assert accumulator_width(383 * 127) == 17 and accumulator_width(0) == 1

    @pytest.mark.parametrize("seed", range(4))
    def test_saturation_reach_is_where_the_law_saturates(self, seed):
        g = np.random.default_rng(seed)
        for _ in range(25):
            m0, shift = int(g.integers(1, 1 << 12)), int(g.integers(0, 14))
            bias, hi = int(g.integers(-(1 << 20), 1 << 20)), int(g.integers(1, 128))
            reach = saturation_reach(m0, bias, shift, hi)
            acc = np.arange(-reach - 3, reach + 4, dtype=np.int64)
            y = apply_law_int64(acc, m0=m0, bias=bias, shift=shift, lo=-hi, hi=hi)
            assert np.all(y[acc > reach] == hi) and np.all(y[acc < -reach] == -hi)
            # tight: the first accumulator that reads hi or the last that reads -hi sits at |acc| = reach
            assert max(abs(int(acc[y == hi].min())), abs(int(acc[y == -hi].max()))) == reach

    def test_law_magnitude_bound_reads_every_column(self):
        assert law_magnitude_bound([10, 10], 3, [0, 1 << 60], 0) == 3 * 11 + (1 << 60)
        assert law_magnitude_bound([10, 1000], [3, 3], [0, 0], [0, 2]) == 3 * 1004

    def test_fp64_bound_is_two_to_the_fifty_three(self):
        assert FP64_EXACT_BOUND == 2 ** 53

    def test_grid_codes_refuse_an_off_grid_value_by_name(self):
        step = torch.tensor(0.25, dtype=torch.float64)
        assert torch.equal(grid_codes(torch.tensor([0.5, -0.75]), step, where="t"), torch.tensor([2.0, -3.0],
                                                                                          dtype=torch.float64))
        with pytest.raises(OffGridInputError, match="off its declared grid"):
            grid_codes(torch.tensor([0.3]), step, where="t")


class TestEntrySnapOneRule:
    def test_exactly_the_tie_witnesses_move(self):
        x = torch.tensor([0.5, -0.5, -2.5, 1.5, 2.5, 0.49, -1.51], dtype=torch.float64)
        snapped = quantize_to_value_grid(x, torch.tensor(7.0, dtype=torch.float64), 4)  # step 1, 7 levels
        assert snapped.tolist() == [1.0, -1.0, -3.0, 2.0, 3.0, 0.0, -2.0]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_every_non_tie_output_is_byte_identical_to_half_to_even(self, dtype):
        g = torch.Generator().manual_seed(8)
        x = (torch.randn(200_000, generator=g) * 40).to(dtype)
        quotient = x / torch.tensor(0.37, dtype=dtype)
        tie = (quotient - torch.trunc(quotient)).abs() == 0.5
        new = round_half_away(quotient)
        old = torch.round(quotient)
        assert new[~tie].numpy().tobytes() == old[~tie].numpy().tobytes()
        assert bool((new[tie].abs() >= old[tie].abs()).all() and ((new[tie] - old[tie]).abs() <= 1).all())
        signed_zero = torch.tensor([-0.0, 0.0, -0.3, 0.3], dtype=dtype)
        assert round_half_away(signed_zero).numpy().tobytes() == torch.round(signed_zero).numpy().tobytes()

    def test_the_model_side_quantizer_snaps_by_the_same_rule(self):
        quantizer = ValueGridQuantizer(4, scale=7.0)
        out = quantizer(torch.tensor([0.5, -2.5], dtype=torch.float64))
        assert out.tolist() == [1.0, -3.0]

    def test_quantize_to_value_grid_rounds_through_the_law_module(self):
        source = (SRC / "models" / "nn" / "activations" / "value_quantizer.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(source))
                  if isinstance(n, ast.FunctionDef) and n.name == "quantize_to_value_grid")
        calls = {ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)}
        assert "round_half_away" in calls and "torch.round" not in calls


class TestOneLawSSOT:
    CONSUMERS = (
        "chip_simulation/value_run/value_law.py",
        "chip_simulation/value_run/int64_golden.py",
        "models/nn/activations/readout_quantizer.py",
        "mapping/export/law_solver.py",
    )
    LAW_NAMES = ("round_half_away", "rnd_shift", "rnd_shift_int64", "apply_law", "apply_law_int64", "adc",
                 "adc_int64")

    def test_executor_golden_quantizer_and_solver_import_the_law_module(self):
        for rel in self.CONSUMERS:
            tree = ast.parse((SRC / rel).read_text())
            sources = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            assert "mimarsinan.models.nn.activations.integer_law" in sources, rel

    def test_no_second_implementation_of_the_law_exists(self):
        offenders = []
        for path in SRC.rglob("*.py"):
            if path.name == "integer_law.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.FunctionDef) and (
                    node.name in self.LAW_NAMES or "half_away" in node.name or node.name.startswith("rnd_")
                ):
                    offenders.append(f"{path.relative_to(SRC)}:{node.name}")
        assert offenders == []
