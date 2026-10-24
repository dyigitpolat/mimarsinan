"""The constant-OFF bias clip on the perceptron's declared input range (D19).

The clip is function-preserving only under two premises: the activation node declares a constant-OFF
floor (it outputs 0 for every pre-activation <= 0; ``models.nn.off_floor``), and every input lies in a
proven range, over which the largest pre-activation is ``sum_i max(w_i lo_i, w_i hi_i)``. The base clip assumed [0, 1] for every
non-encoding perceptron, so on a value-domain target it moved the bias of an affine stem reading the
signed normalized image (R10 w1: stem channel 41 from -3 to -1 weight steps, 780 of 262,144 stem
outputs of 4 images changed).

New API names are imported inside the tests so that, on the base tree, the behavioral tests fail on
their assertions (the defect) rather than all at import.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer, LeakyGradReLU
from mimarsinan.models.nn.activations.value_quantizer import BoundaryGrid, ValueGridQuantizer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron, as_affine_package
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.perceptron.bias_saturation import clip_off_saturated_effective_bias
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

GRID_SCALE = 2.024258
"""The R10 w1 stem's calibrated 5-bit boundary grid (probe_wq_entry_R10w100_b0.log)."""

SPECIMEN_BIAS = -0.670730
"""R10 w1 stem channel 41's effective bias at WQ entry (unit bound -0.362253, grid bound -2.942169)."""

WEIGHTS = torch.tensor([
    [1.6, -0.3, 0.2, 0.1, -0.4, 0.0, 0.3, -0.2],  # max|w| 1.6 sets the 4-bit weight grid S = floor(7 / 1.6) = 4
    [0.2, 0.162253, -0.5, -0.5912, 0.0, 0.0, 0.0, 0.0],  # the specimen: positive reach 0.362253
    [0.5, 0.5, 0.1, -0.1, 0.2, 0.0, 0.0, 0.1],
])


def _specimen(*, affine: bool, grid_scale: "float | None" = GRID_SCALE, bias: float = SPECIMEN_BIAS) -> Perceptron:
    """A perceptron with the specimen weights; ``affine`` = the value-domain package (Identity activation)."""
    p = Perceptron(3, 8, normalization=nn.Identity())
    if affine:
        as_affine_package(p)
    else:
        p.activation = LeakyGradReLU()
    with torch.no_grad():
        p.layer.weight.copy_(WEIGHTS)
        p.layer.bias.copy_(torch.tensor([0.05, bias, 0.1]))
    if grid_scale is not None:
        quantizer = ValueGridQuantizer(5)
        quantizer.calibrate(grid_scale)
        p.append_input_wire_op(quantizer)
    return p.eval()


def _effective_bias(p) -> torch.Tensor:
    return PerceptronTransformer().get_effective_bias(p).detach().clone()


def _signed_inputs(n: int = 4096, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 8, generator=g) * 2.0 - 1.0) * 1.5 * GRID_SCALE  # the wire saturates beyond its grid


def _grid_bound(weights: torch.Tensor, scale: float) -> torch.Tensor:
    return -(weights.abs() * scale).sum(dim=1)


class TestValueDomainAffinePerceptron:
    """An affine package has no OFF state: no bias move is function-preserving on it."""

    def test_signed_input_channel_keeps_its_bias(self):
        p = _specimen(affine=True)
        before = _effective_bias(p)
        assert clip_off_saturated_effective_bias(p) == 0
        assert torch.equal(_effective_bias(p), before)

    def test_signed_input_forward_is_unchanged(self):
        p = _specimen(affine=True)
        x = _signed_inputs()
        with torch.no_grad():
            before = p(x).clone()
        clip_off_saturated_effective_bias(p)
        with torch.no_grad():
            assert torch.equal(p(x), before)

    def test_channel_off_on_its_whole_range_is_still_not_clipped(self):
        p = _specimen(affine=True, bias=-4.0)  # below the grid bound: <= 0 everywhere, yet the output is linear in b
        before = _effective_bias(p)
        assert clip_off_saturated_effective_bias(p) == 0
        assert torch.equal(_effective_bias(p), before)

    def test_float_boundary_without_a_grid_is_not_clipped(self):
        for grid_scale in (None, 0.0):  # no wire op, and an unarmed (identity) grid
            p = _specimen(affine=True, grid_scale=grid_scale)
            before = _effective_bias(p)
            assert clip_off_saturated_effective_bias(p) == 0, grid_scale
            assert torch.equal(_effective_bias(p), before), grid_scale

    def test_napq_projection_keeps_the_specimen_at_minus_three_weight_steps(self):
        p = _specimen(affine=True)
        NormalizationAwarePerceptronQuantization(4, "cpu", rate=1.0, two_scale=True).transform(p)
        weight_scale = float(p.parameter_scale)
        bias_scale = float(p.bias_scale)
        assert (weight_scale, bias_scale) == (4.0, 4.0)
        steps = _effective_bias(p) * weight_scale
        assert float(steps[1]) == -3.0, steps


class TestFlooredPerceptronOnADeclaredRange:
    """A floored perceptron whose wire declares a signed grid: the bound is taken on that grid."""

    def test_channel_on_for_signed_inputs_keeps_its_bias(self):
        p = _specimen(affine=False)
        before = _effective_bias(p)
        x = _signed_inputs()
        with torch.no_grad():
            out_before = p(x).clone()
        assert clip_off_saturated_effective_bias(p) == 0
        assert torch.equal(_effective_bias(p), before)
        with torch.no_grad():
            assert torch.equal(p(x), out_before)

    def test_channel_provably_off_on_the_declared_range_is_clipped_to_its_bound(self):
        p = _specimen(affine=False, bias=-4.0)
        x = _signed_inputs()
        with torch.no_grad():
            out_before = p(x).clone()
        assert clip_off_saturated_effective_bias(p) == 1
        bound = _grid_bound(WEIGHTS, GRID_SCALE)
        assert torch.isclose(_effective_bias(p)[1], bound[1], atol=1e-5)
        with torch.no_grad():
            assert torch.equal(p(x), out_before)

    def test_declared_range_clip_is_idempotent(self):
        p = _specimen(affine=False, bias=-4.0)
        assert clip_off_saturated_effective_bias(p) == 1
        w, b = p.layer.weight.data.clone(), p.layer.bias.data.clone()
        assert clip_off_saturated_effective_bias(p) == 0
        assert torch.equal(p.layer.weight.data, w)
        assert torch.equal(p.layer.bias.data, b)

    def test_grid_not_at_the_wire_output_declares_no_range(self):
        class Gain(nn.Module):
            def forward(self, x):
                return 3.0 * x

        p = _specimen(affine=False, bias=-4.0)
        p.append_input_wire_op(Gain())  # the grid no longer bounds what the layer reads
        before = _effective_bias(p)
        assert clip_off_saturated_effective_bias(p) == 0
        assert torch.equal(_effective_bias(p), before)

    def test_unarmed_grid_declares_no_range(self):
        p = _specimen(affine=False, bias=-4.0, grid_scale=0.0)
        before = _effective_bias(p)
        assert clip_off_saturated_effective_bias(p) == 0
        assert torch.equal(_effective_bias(p), before)

    def test_raw_domain_bound_is_invariant_to_per_input_scales(self):
        plain = _specimen(affine=False, bias=-4.0)
        scaled = _specimen(affine=False, bias=-4.0)
        scaled.per_input_scales = torch.tensor([2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0])
        clip_off_saturated_effective_bias(plain)
        clip_off_saturated_effective_bias(scaled)
        assert torch.isclose(scaled.layer.bias.data[1], plain.layer.bias.data[1], atol=1e-5)
        assert torch.isclose(plain.layer.bias.data[1], _grid_bound(WEIGHTS, GRID_SCALE)[1], atol=1e-5)

    def test_encoding_layer_with_a_grid_is_skipped(self):
        p = _specimen(affine=False, bias=-4.0)
        p.is_encoding_layer = True
        assert clip_off_saturated_effective_bias(p) == 0


@dataclass(frozen=True)
class _OffsetGrid(BoundaryGrid):
    """A grid declared with an entry offset and code range (the shape of the lanes unit's declared grid)."""

    offset: float = 0.0
    code_min: int = 0

    def entry_offset(self) -> float:
        return float(self.offset)

    @property
    def code_range(self) -> Tuple[int, int]:
        return int(self.code_min), self.levels


class _DeclaredQuantizer(ValueGridQuantizer):
    """A boundary quantizer holding a declared grid object in place of its calibrated scale."""

    def __init__(self, grid: BoundaryGrid) -> None:
        super().__init__(grid.bits, grid.scale)
        self._grid = grid

    @property
    def grid(self) -> BoundaryGrid:
        return self._grid


class TestBoundaryGridValueRange:
    def test_symmetric_grid_value_range(self):
        grid = BoundaryGrid(scale=2.0, bits=5)
        assert grid.value_range == (-15 * grid.step, 15 * grid.step)

    def test_value_range_follows_entry_offset_and_code_range(self):
        grid = _OffsetGrid(scale=1.5, bits=5, offset=-1.0, code_min=0)
        assert grid.value_range == (0 * grid.step + 1.0, 15 * grid.step + 1.0)

    def test_wire_output_grid_reads_the_last_op(self):
        from mimarsinan.models.nn.activations.value_quantizer import wire_output_grid

        q = ValueGridQuantizer(5)
        q.calibrate(1.0)
        assert wire_output_grid(q) == q.grid
        assert wire_output_grid(nn.Sequential(ChipInputQuantizer(8, 1.0), q)) == q.grid
        assert wire_output_grid(nn.Sequential(q, nn.Tanh())) is None
        assert wire_output_grid(ValueGridQuantizer(5)) is None  # unarmed: the identity
        assert wire_output_grid(nn.Identity()) is None

    def test_declared_range_is_hulled_with_zero_for_padding(self):
        grid = _OffsetGrid(scale=1.5, bits=5, offset=-1.0, code_min=0)  # values in [1.0, 2.5]: excludes 0
        p = _specimen(affine=False, grid_scale=None, bias=-40.0)
        p.append_input_wire_op(_DeclaredQuantizer(grid))
        assert clip_off_saturated_effective_bias(p) == 1
        high = grid.value_range[1]
        hulled = -torch.maximum(WEIGHTS * 0.0, WEIGHTS * high).sum(dim=1)
        assert torch.isclose(p.layer.bias.data[1], hulled[1], atol=1e-5)


class TestLanesDeclaredBoundaryGrid:
    """The D19 and lanes seam (integration D19 conflict row 2): the lanes unit's ``DeclaredBoundaryGrid`` (WI-5a),
    installed by ``ValueGridQuantizer.declare``, is the declared input range the clip reads, through its own
    ``code_range``, ``step`` and ``entry_offset()`` overrides (moved here from the integration's seam script)."""

    SEAM_WEIGHTS = torch.tensor([[0.5, -0.25, 0.25, -0.5], [0.3, 0.2, -0.1, 0.4]])

    def _floored(self, grid, bias=-40.0, affine=False) -> Perceptron:
        p = Perceptron(2, 4, normalization=nn.Identity())
        if affine:
            as_affine_package(p)
        else:
            p.activation = LeakyGradReLU()
        with torch.no_grad():
            p.layer.weight.copy_(self.SEAM_WEIGHTS)
            p.layer.bias.copy_(torch.tensor([bias, 0.1]))
        quantizer = ValueGridQuantizer(grid.bits)
        quantizer.declare(grid)
        p.append_input_wire_op(quantizer)
        return p.eval()

    def test_declared_grid_value_range_follows_its_offset_and_codes(self):
        from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid

        g = DeclaredBoundaryGrid(scale=1.5, bits=5, offset=0.5, code_min=-3, code_max=12)
        assert g.value_range == (-3 * g.step - 0.5, 12 * g.step - 0.5)
        u = DeclaredBoundaryGrid(scale=3.1, bits=4, offset=0.0, code_min=0, code_max=15, unsigned_dac=True)
        assert u.unsigned and u.value_range == (0.0, 15 * (3.1 / 15))

    def test_the_clip_reads_the_declared_grid_hulled_with_zero(self):
        from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid, wire_output_grid
        from mimarsinan.transformations.perceptron.bias_saturation import declared_input_range

        g = DeclaredBoundaryGrid(scale=1.5, bits=5, offset=-1.0, code_min=0, code_max=15)  # values [1.0, 2.5]
        p = self._floored(g)
        assert wire_output_grid(p.input_activation) is g
        _low, high = g.value_range
        rng = declared_input_range(p)
        assert rng is not None and float(rng.low) == 0.0 and float(rng.high) == high  # hulled with 0
        assert clip_off_saturated_effective_bias(p) == 1
        expected = -torch.maximum(self.SEAM_WEIGHTS * 0.0, self.SEAM_WEIGHTS * high).sum(dim=1)
        assert torch.isclose(p.layer.bias.data[0], expected[0], atol=1e-6)

    def test_an_affine_reader_of_a_declared_grid_is_never_clipped(self):
        from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid

        g = DeclaredBoundaryGrid(scale=1.5, bits=5, offset=0.0, code_min=-15, code_max=15)
        p = self._floored(g, affine=True)
        before = _effective_bias(p).clone()
        assert clip_off_saturated_effective_bias(p) == 0
        assert torch.equal(_effective_bias(p), before)


class TestOffBoundOnARange:
    def test_bound_is_the_largest_preactivation_over_the_box(self):
        from mimarsinan.transformations.perceptron.bias_saturation import InputRange, off_saturation_bias_bound

        g = torch.Generator().manual_seed(3)
        w = torch.randn(4, 5, generator=g, dtype=torch.float64)
        low = -torch.rand(1, 5, generator=g, dtype=torch.float64)
        high = torch.rand(1, 5, generator=g, dtype=torch.float64)
        bound = off_saturation_bias_bound(w, InputRange(low, high))
        corners = torch.tensor(list(itertools.product([0, 1], repeat=5)), dtype=torch.float64)
        x = low + corners * (high - low)  # every corner of the box
        assert torch.allclose(-bound, (x @ w.T).max(dim=0).values)

    def test_unit_range_is_bitwise_the_legacy_formula(self):
        from mimarsinan.transformations.perceptron.bias_saturation import UNIT_INPUT_RANGE, off_saturation_bias_bound

        g = torch.Generator().manual_seed(4)
        for shape in ((6, 9), (3, 2, 3, 3)):
            w = torch.randn(*shape, generator=g)
            legacy = -w.flatten(1).clamp(min=0).sum(dim=1)
            assert torch.equal(off_saturation_bias_bound(w), legacy)
            assert torch.equal(off_saturation_bias_bound(w, UNIT_INPUT_RANGE), legacy)


class TestEventDomainUnchanged:
    """Floored perceptrons without a value-domain wire keep the event wire's [0, 1] premise, byte for byte."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("wire", ["none", "chip_input"])
    def test_clip_equals_the_legacy_unit_clip(self, seed, wire):
        torch.manual_seed(seed)
        p = Perceptron(5, 12, normalization=nn.Identity())
        p.activation = LeakyGradReLU()
        if wire == "chip_input":
            p.append_input_wire_op(ChipInputQuantizer(8, 1.0))
        with torch.no_grad():
            p.layer.bias.data[seed % 5] = -20.0
        w = PerceptronTransformer().get_effective_weight(p).detach()
        b = PerceptronTransformer().get_effective_bias(p).detach()
        legacy_bound = -w.flatten(1).clamp(min=0).sum(dim=1)
        expected = torch.where(b < legacy_bound, legacy_bound, b) * p.activation_scale.data
        assert clip_off_saturated_effective_bias(p) >= 1
        assert torch.equal(p.layer.bias.data, expected)
