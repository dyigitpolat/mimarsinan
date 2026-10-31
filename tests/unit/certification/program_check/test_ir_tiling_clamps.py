"""[U-CR C-1, SD-06 SDB-R2] The ir-tiling law rule reads two column-aligned laws as one function when they differ only
in clamp fields (the ADC's ``adc_levels``, the output clamp's ``lo`` and ``hi``) and, on every column where a clamp
differs, both clamps contain that column's reachable bound: the ADC's at ``rnd_t(acc_bound)``, the output's at
``rnd_s(|M0| (acc_bound + 2^t) + |B|)`` (``law_magnitude_bound`` per column), with ``acc_bound`` the predicate's own
``column_bounds``. The identity output law clamps at a core-wide magnitude bound that never binds, so a tile of an
output layer wider than the core carries other ``lo`` and ``hi`` than the unbounded neuron (SD-06: C1 B5i and C3 B5m
at 256x64 read Theorem 1 'failed: ir-tiling' on exact programs). A clamp that binds on a reachable accumulator, and an
operand column (whose term the accumulator bound does not cover), still declare another function by name."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.program_check import CheckDeclaration
from mimarsinan.certification.program_check.report import ESTABLISHED, ConditionResult
from mimarsinan.certification.program_check.unbounded import check_ir_tiling
from mimarsinan.certification.program_check.unbounded.laws import same_function
from mimarsinan.mapping.export.law_grids import CoreGrid
from mimarsinan.mapping.export.output_law import solve_identity_law
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.ir.lane_law import LaneLaw
from mimarsinan.models.nn.activations.integer_law import apply_law_int64, law_magnitude_bound
from unit.mapping.declared_programs import build_declared_program


def _identity_pair():
    """SD-06's fixture: a 100-column identity output law and its 64-column tile, the widest column (80) outside it."""
    rng = np.random.default_rng(0)
    bounds = [int(v) for v in rng.integers(1000, 3046, size=100)]
    bounds[80] = 4000
    bias = rng.normal(size=100) * 0.05
    grid = CoreGrid(input_step=0.01, input_levels=15, output_step=0.01, output_range=1.0)
    spec = SimpleNamespace(mult_bits=24, adc_bits=32)
    whole = solve_identity_law(core="classifier", theta=1.0, bias=bias, acc_bounds=bounds, grid=grid, spec=spec)
    tile = solve_identity_law(core="classifier_tile_0_64", theta=1.0, bias=bias[:64], acc_bounds=bounds[:64],
                              grid=grid, spec=spec)
    return tile, whole.take_columns(np.arange(64)), np.asarray(bounds[:64], dtype=np.float64)


def _outputs(law, bounds):
    """Each column's law over every reachable accumulator (|acc| at most its bound)."""
    out = []
    for j, bound in enumerate(bounds.astype(np.int64).tolist()):
        acc = np.arange(-bound, bound + 1, dtype=np.int64)
        out.append(apply_law_int64(acc, m0=law.m0[j], bias=law.bias[j], shift=law.shift[j], lo=law.lo[j],
                                   hi=law.hi[j], adc_shift=law.adc_shift[j], adc_levels=law.adc_levels[j]))
    return out


def _equal_outputs(a, b, bounds) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(_outputs(a, bounds), _outputs(b, bounds)))


class TestSameFunction:
    def test_the_tile_of_an_identity_output_law_declares_the_unbounded_neurons_function(self):
        tile, whole, bounds = _identity_pair()
        moved = [f for f in ("input_step", "m0", "shift", "bias", "lo", "hi", "output_step", "adc_shift", "adc_levels")
                 if not np.array_equal(getattr(tile, f), getattr(whole, f))]
        assert moved == ["lo", "hi"]
        assert _equal_outputs(tile, whole, bounds)
        assert same_function(tile, whole, bounds, bounds)

    @pytest.mark.parametrize("field", ["hi", "lo"])
    def test_a_clamp_that_binds_on_a_reachable_accumulator_still_differs(self, field):
        tile, whole, bounds = _identity_pair()
        top = int(tile.m0[5]) * int(bounds[5])          # |M0 acc| at the column's accumulator bound
        values = np.asarray(getattr(tile, field)).copy()
        values[5] = top + int(tile.bias[5]) - 1 if field == "hi" else -top + int(tile.bias[5]) + 1
        bound = replace(tile, **{field: values})
        assert not _equal_outputs(bound, whole, bounds)   # acc = +-bound reads the clamp
        assert not same_function(bound, whole, bounds, bounds)

    def test_a_clamp_inside_the_reachable_outputs_reads_two_functions(self):
        tile, whole, bounds = _identity_pair()
        hi = np.asarray(tile.hi).copy()
        hi[3] = 7
        bound = replace(tile, hi=hi)
        assert not _equal_outputs(bound, whole, bounds)
        assert not same_function(bound, whole, bounds, bounds)

    def test_an_undeclared_row_range_leaves_the_clamps_compared_exactly(self):
        tile, whole, bounds = _identity_pair()
        unknown = bounds.copy()
        unknown[0] = np.nan
        assert not same_function(tile, whole, unknown, unknown)

    def test_a_shifted_law_reads_its_clamp_against_the_rounded_reach(self):
        """Every clamp of every law: at shift s the output clamp holds ``rnd_s`` of the |z| bound."""
        law = CoreLaw(input_step=[0.1] * 2, m0=[3, 5], shift=[2, 3], bias=[4, -7], lo=[-10, -10], hi=[10, 10],
                      output_step=[0.2] * 2, adc_shift=[0, 0], adc_levels=[100, 100])
        bounds = np.array([9.0, 6.0])
        reach = [law_magnitude_bound(9, 3, 4, 0), law_magnitude_bound(6, 5, -7, 0)]   # 34 and 42
        rounded = [(r + (1 << (s - 1))) >> s for r, s in zip(reach, (2, 3))]          # 9 and 5
        wide = replace(law, lo=[-r for r in rounded], hi=rounded)
        wider = replace(law, lo=[-50, -60], hi=[50, 60])
        assert _equal_outputs(wide, wider, bounds) and same_function(wide, wider, bounds, bounds)
        narrow = replace(law, hi=[7, rounded[1]])              # column 0 reaches rnd_2(31) = 8
        assert not _equal_outputs(narrow, wider, bounds) and not same_function(narrow, wider, bounds, bounds)

    def test_a_narrow_adc_reads_its_levels_against_the_rounded_accumulator_bound(self):
        """The ADC's levels at step 2^t hold ``rnd_t(acc_bound)`` codes; one level fewer can bind."""
        law = CoreLaw(input_step=[0.1], m0=[1], shift=[0], bias=[0], lo=[-1000], hi=[1000], output_step=[0.2],
                      adc_shift=[2], adc_levels=[3])
        bounds = np.array([10.0])
        held = replace(law, adc_levels=[(10 + 2) >> 2])        # 3 codes reach 12, beyond 10
        more = replace(law, adc_levels=[63])
        assert _equal_outputs(held, more, bounds) and same_function(held, more, bounds, bounds)
        fewer = replace(law, adc_levels=[2])
        assert not _equal_outputs(fewer, more, bounds) and not same_function(fewer, more, bounds, bounds)

    def test_an_operand_column_keeps_its_clamps_compared_exactly(self):
        """A lane law's fused operand enters before the clamp, and the accumulator bound does not cover it."""
        tile, whole, bounds = _identity_pair()
        rows = np.full(64, -1)
        rows[0] = 3
        a = LaneLaw.of(tile, operand_row=rows, operand_m=np.where(rows >= 0, 2, 0))
        b = LaneLaw.of(whole, operand_row=rows, operand_m=np.where(rows >= 0, 2, 0))
        assert not same_function(a, b, bounds, bounds)
        plain_a, plain_b = LaneLaw.of(tile), LaneLaw.of(whole)
        assert same_function(plain_a, plain_b, bounds, bounds)

    def test_a_field_other_than_a_clamp_still_reads_another_function(self):
        tile, whole, bounds = _identity_pair()
        assert not same_function(replace(tile, bias=np.asarray(tile.bias) + 1), whole, bounds, bounds)


def _head() -> nn.Module:
    torch.manual_seed(11)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 12)).eval()


@pytest.fixture(scope="module")
def head_irs(tmp_path_factory):
    """A 12-class identity output layer tiled at 8 neurons per core, and the same network at unbounded geometry."""
    model = _head()
    path = tmp_path_factory.mktemp("ucr_clamps")
    tiled, _p, _d = build_declared_program(copy.deepcopy(model), (8,), num_classes=12, state_path=path / "t.json",
                                           max_axons=64, max_neurons=8)
    unbounded, _p, _d = build_declared_program(copy.deepcopy(model), (8,), num_classes=12,
                                               state_path=path / "u.json")
    return tiled.ir, unbounded.ir


_DECLARATION = CheckDeclaration(semantics="value_integer", window=1, pass_transfer="collapse",
                                accumulation={"value_census": "float64", "int64_golden": "int64"},
                                upload={"value_census": "float32"})


def _ir_tiling(tiled, unbounded) -> ConditionResult:
    result = ConditionResult("ir-tiling", ESTABLISHED)
    check_ir_tiling(tiled, unbounded, declaration=_DECLARATION, result=result)
    return result


def _output_tiles(ir):
    last = max(int(n.perceptron_index) for n in ir.get_neural_cores())
    return [n for n in ir.get_neural_cores() if int(n.perceptron_index) == last]


class TestTiledIdentityOutputLayer:
    def test_the_tiles_carry_other_clamps_than_the_unbounded_neuron(self, head_irs):
        tiled, unbounded = head_irs
        (whole,) = _output_tiles(unbounded)
        tiles = _output_tiles(tiled)
        assert len(tiles) == 2
        assert any(int(t.readout_law.hi[0]) != int(whole.readout_law.hi[0]) for t in tiles)

    def test_the_tiled_identity_output_layer_is_established(self, head_irs):
        result = _ir_tiling(*head_irs)
        assert not result.violations, [v.text() for v in result.violations]
        assert "with its fan-in, function" in result.note

    def test_a_tile_whose_clamp_binds_still_fails_by_name(self, head_irs):
        tiled, unbounded = head_irs
        mutated = copy.deepcopy(tiled)
        tile = _output_tiles(mutated)[0]
        hi = np.asarray(tile.readout_law.hi).copy()
        hi[0] = 1
        tile.readout_law = replace(tile.readout_law, hi=hi)
        result = _ir_tiling(mutated, unbounded)
        assert "law" in {v.rule for v in result.violations}, [v.text() for v in result.violations]
