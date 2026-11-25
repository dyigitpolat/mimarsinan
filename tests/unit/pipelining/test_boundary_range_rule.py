"""[U-BRANGE] The boundary range rule: each boundary's scale is the scale of least squared error of its own grid.

W38 group 6's diagnosis of record (ViT-B/16, arm W): Boundary Quantization set each boundary's one scale to the 0.999
quantile of |x| pooled over every position, so two values a LayerNorm writes at one position of every input sat in the
tail the quantile discards and were clamped on every input. The rule weighs a value by its error, never by its rarity:
the scale is the least squared error of the boundary's own grid (``quantize_to_value_grid`` at the quantizer's bits)
over the calibration batches, rounding and clamp both counted, searched [U-BRANGE2 (c)] over the whole interval up to the
calibration maximum, so it clamps where the error minimum says so, at any bit width and any magnitude.

T-1: the failing form in miniature (its bound named; the squared-error clause decides). T-2: the rule's properties on
synthetic seams at 4 and 8 bits, with no floor, the unexercised-boundary refusal and a declared grid kept. T-3: the
grid and the certificates' formula unchanged (only the scale's value moves). Every error here is read directly from
the grid (``range_rule_reference``), never from the toolchain's own search.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.value_quantizer import BoundaryGrid, DeclaredBoundaryGrid, quantize_to_value_grid
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from unit.pipelining.range_rule_reference import (
    RELATIVE,
    grid_error,
    interval,
    reference_errors,
    signed_grid,
)

# T-1's form: a host LayerNorm over CHANNELS feeding one Linear perceptron; token 0 carries, on every sample, one
# channel the LayerNorm writes at about 25 (7.6 times the bulk's 0.999 quantile of about 3.3), one value of
# TOKENS * CHANNELS per sample (6.1e-5 of the seam's entries, inside the 1e-3 tail the quantile discards).
CHANNELS, TOKENS, SAMPLES, OUTPUTS, HOT = 1024, 16, 16, 10, 259


class _LayerNormForm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.norm = nn.LayerNorm(CHANNELS)
        self.entry = Perceptron(OUTPUTS, CHANNELS)

    def forward(self, x):
        return self.entry(self.norm(x))


def _tokens(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SAMPLES, TOKENS, CHANNELS, generator=generator)
    x[:, 0, HOT] = -40.0 - torch.rand(SAMPLES, generator=generator)
    return x


def _calibrated_form(bits: int = 8):
    form = _LayerNormForm().eval()
    batches = [_tokens(1), _tokens(2)]
    with torch.no_grad():
        seams = [form.norm(x) for x in batches]
    (quantizer,) = install_boundary_quantizers([form.entry], bits)
    census = calibrate_boundary_scales(form, [form.entry], [quantizer], batches)
    return form, quantizer, seams, census


class TestT1TheFailingFormInMiniature:
    """The deciding assertion is the squared-error clause (the diagnosis check's c-m3); the output bound is named."""

    def test_the_form_has_the_systematic_value_inside_the_discarded_tail(self):
        _form, _q, seams, _census = _calibrated_form()
        low, _high = interval([s.abs() for s in seams])
        hot = torch.cat([s[:, 0, HOT].abs() for s in seams])
        assert float(hot.min()) > 7.0 * low  # about 7.6 times the pooled 0.999 quantile, on every sample
        assert 1.0 / (TOKENS * CHANNELS) < 1e-3  # one value per sample sits inside the discarded tail

    def test_the_scale_s_squared_error_is_at_most_the_quantile_s_and_the_maximum_s(self):
        _form, quantizer, seams, _census = _calibrated_form()
        magnitudes = [s.abs() for s in seams]
        low, high = interval(magnitudes)
        snap = signed_grid(8)
        chosen = grid_error(magnitudes, quantizer.grid.scale, snap)
        assert chosen <= grid_error(magnitudes, low, snap) * (1 + RELATIVE)
        assert chosen <= grid_error(magnitudes, high, snap) * (1 + RELATIVE)
        assert chosen < grid_error(magnitudes, low, snap)  # strictly better than the discarded tail's clamp

    # [U-BRANGE2 stage 1d, the check's n-2] the reference reads the grid's float64 error at every candidate (45 s on a
    # shared-server worker at load 180, over the default 60 s budget under heavier load): a time budget of its own
    @pytest.mark.timeout(240)
    def test_the_value_is_kept_or_its_clamp_is_the_error_minimum(self):
        _form, quantizer, seams, _census = _calibrated_form()
        magnitudes = [s.abs() for s in seams]
        hot = max(float(s[:, 0, HOT].abs().max()) for s in seams)
        scale = quantizer.grid.scale
        _scales, errors = reference_errors(magnitudes, signed_grid(8))
        kept = scale >= hot
        assert kept or grid_error(magnitudes, scale, signed_grid(8)) <= min(errors) * (1 + RELATIVE)

    def test_token_0_agrees_with_float_within_the_rounding_bound(self):
        """The named bound: per output, ``rounding_bound = sum_i |W_ji| * step / 2`` at the boundary's own step, the
        largest departure the grid's rounding alone can make (no clamp)."""
        form, quantizer, seams, _census = _calibrated_form()
        weight = form.entry.layer.weight.detach().double()
        step = quantizer.grid.step
        rounding_bound = weight.abs().sum(dim=1) * (step / 2.0)
        for seam in seams:
            x = seam[:, 0, :].double()
            departure = (quantize_to_value_grid(x, quantizer.grid.scale, 8) - x) @ weight.T
            assert bool((departure.abs() <= rounding_bound).all()), float((departure.abs() / rounding_bound).max())


def _laplace(generator, n):
    u = torch.rand(n, generator=generator) - 0.5
    return -torch.sign(u) * torch.log1p(-2.0 * u.abs())


_DISTRIBUTIONS = {
    "uniform": lambda generator, n: torch.rand(n, generator=generator) * 2.0 - 1.0,
    "gaussian": lambda generator, n: torch.randn(n, generator=generator),
    "laplace": _laplace,
}


def _seam_batches(distribution: str, tail: bool, seed: int = 3):
    """Two batches of (samples, tokens, channels) = (16, 64, 64); with ``tail`` one position of every sample (token
    0, channel 3: 1/4,096 of the entries) at 8 times the bulk's 0.999 quantile."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(2):
        x = _DISTRIBUTIONS[distribution](generator, 16 * 64 * 64).reshape(16, 64, 64)
        batches.append(x)
    if tail:
        bulk = max(float(torch.quantile(b.abs().reshape(-1), 0.999)) for b in batches)
        for b in batches:
            b[:, 0, 3] = 8.0 * bulk
    return batches


def _calibrate(batches, bits: int):
    perceptron = Perceptron(4, 64)
    (quantizer,) = install_boundary_quantizers([perceptron], bits)
    calibrate_boundary_scales(perceptron, [perceptron], [quantizer], batches)
    return quantizer.grid.scale


_CASES = [(d, t, b) for d in _DISTRIBUTIONS for t in (False, True) for b in (4, 8)]


class TestT2TheRuleSProperties:
    @pytest.mark.parametrize("distribution, tail, bits", _CASES)
    def test_the_chosen_error_is_at_most_the_quantile_s_and_the_maximum_s(self, distribution, tail, bits):
        batches = _seam_batches(distribution, tail)
        magnitudes = [b.abs() for b in batches]
        scale = _calibrate(batches, bits)
        low, high = interval(magnitudes)
        snap = signed_grid(bits)
        assert high * 2.0 ** -12 * (1 - RELATIVE) <= scale <= high * (1 + RELATIVE)  # [U-BRANGE2 (c)] the whole interval
        chosen = grid_error(magnitudes, scale, snap)
        assert chosen <= grid_error(magnitudes, low, snap) * (1 + RELATIVE)
        assert chosen <= grid_error(magnitudes, high, snap) * (1 + RELATIVE)

    @pytest.mark.parametrize("distribution, tail, bits", _CASES)
    def test_the_chosen_scale_is_the_least_error_of_the_search(self, distribution, tail, bits):
        batches = _seam_batches(distribution, tail)
        magnitudes = [b.abs() for b in batches]
        scale = _calibrate(batches, bits)
        _scales, errors = reference_errors(magnitudes, signed_grid(bits))
        assert grid_error(magnitudes, scale, signed_grid(bits)) <= min(errors) * (1 + RELATIVE)

    @pytest.mark.parametrize("distribution", sorted(_DISTRIBUTIONS))
    def test_the_choice_is_deterministic(self, distribution):
        batches = _seam_batches(distribution, True)
        assert _calibrate(batches, 8) == _calibrate([b.clone() for b in batches], 8)

    @pytest.mark.parametrize("distribution", sorted(_DISTRIBUTIONS))
    def test_fewer_bits_clamp_at_least_as_much(self, distribution):
        batches = _seam_batches(distribution, True)
        assert _calibrate(batches, 4) <= _calibrate(batches, 8)

    def test_tiny_magnitudes_take_the_rule_s_range_with_no_floor(self):
        perceptron = Perceptron(4, 4)
        (quantizer,) = install_boundary_quantizers([perceptron], 8)
        calibrate_boundary_scales(perceptron, [perceptron], [quantizer], [torch.full((8, 4), 1e-7)])
        assert quantizer.grid.scale == pytest.approx(1e-7, rel=1e-6)  # [U-BRANGE2] the least error, no floor

    def test_an_unexercised_boundary_is_refused(self):
        reached, dead = Perceptron(4, 4), Perceptron(4, 4)
        quantizers = install_boundary_quantizers([reached, dead], 8)
        with pytest.raises(RuntimeError, match="every magnitude .* is 0"):
            calibrate_boundary_scales(reached, [reached, dead], quantizers, [torch.randn(16, 4)])

    def test_a_declared_grid_still_replaces_the_calibrated_one(self, monkeypatch, tmp_path):
        from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
        from unit.mapping.declared_programs import write_state
        from unit.pipelining.test_declared_grid_steps import _boundary_step, _flow

        flow = _flow()
        declaration = write_state(tmp_path / "s.json", {
            "_0": {"entry": {"scale": 6.0, "offset": 3.0, "rounding": "floor", "codes": [0, 15]}}})
        _boundary_step(monkeypatch, flow, quantizer_state=declaration)
        names = perceptron_fx_names(flow.get_mapper_repr())
        declared = names["_0"].input_activation.grid
        assert isinstance(declared, DeclaredBoundaryGrid) and declared.scale == 6.0
        assert type(names["_2"].input_activation.grid) is BoundaryGrid


class TestT3TheGridAndItsCertificatesUnchanged:
    def test_the_census_errors_are_the_grid_s_own_squared_error(self):
        """One formula: the rule's errors at the chosen scale and at both ends of its search equal the squared error
        of ``quantize_to_value_grid`` itself on the seam's values."""
        _form, quantizer, seams, census = _calibrated_form()
        assert census is not None and len(census) == 1
        row = census[0]
        magnitudes = [s.abs() for s in seams]
        snap = signed_grid(8)
        for scale, error in ((row.scale, row.squared_error), (row.low_span, row.low_span_error),
                             (row.maximum, row.maximum_error)):
            direct = grid_error(magnitudes, scale, snap)
            assert abs(error - direct) <= RELATIVE * direct, (scale, error, direct)
        assert row.scale == quantizer.grid.scale

    def test_the_installed_snap_is_the_grid_at_the_chosen_scale(self):
        """The forward is the STE ``x + (snap(x) - x).detach()``: the grid's values up to that sum's rounding."""
        form, quantizer, seams, _census = _calibrated_form()
        x = seams[0]
        with torch.no_grad():
            snapped = quantizer(x)
        torch.testing.assert_close(snapped, quantize_to_value_grid(x, quantizer.scale, 8), rtol=0.0, atol=1e-5)

    @pytest.mark.parametrize("name", ["mlp", "lenet5"])
    def test_nf_equals_scm_equals_hcm_at_atol_zero_at_the_rule_s_scales(self, name):
        from unit.chip_simulation.value_run.integer_programs import INSTANCES, assert_exact, integer_inputs

        build, packed = INSTANCES[name]
        program = build()
        x = integer_inputs(program, 32).double()
        with torch.no_grad():
            scm = program.identity()(x)
            assert_exact(program.nf()(x), scm)
            assert_exact(program.packed(**packed)(x), scm)
