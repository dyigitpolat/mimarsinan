"""[U-WQB A, phase C defect D-WQ] The two-scale projection's bias on an on-chip bias register.

The two-scale projection derives the weight grid ``s_w`` from ``max|w|`` alone. On a param-encoded platform the bias
rides ``r`` always-on rows of ±q_max weights (bias-row splitting), so its grid is ``s_w / r`` with
``r = ceil(max|b| s_w / q_max)``. On a platform whose cores carry an on-chip bias register the WQ step routed the bias
to the same coarse grid although the export already allocates the register ``±q_max r`` in weight-grid units: the
bias was rounded to multiples of ``r`` weight steps (``r`` 2 to 28 on the T-LIF rows, 0.3 to 1.9 spike counts of error
per neuron and window at 4 bits; ``RA/12_tcad_readiness/phase_c_r/d_wq/report.md`` sections 2 and 3).

On the register route the bias is now coded on the weight lattice itself, ``round(b s_w)``, within the span
``[-(q_max+1) r, q_max r]`` the export allocates (it never binds at projection: ``|b s_w| <= q_max r`` by the
definition of ``r``), and ``bias_scale == parameter_scale``. The param-encoded route keeps its equal-share ``r``-multiple
grid byte for byte. The projection is read as the WQ step's tuner applies it, so a tree without the route (R) reads
its own projection.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import WeightQuantizationStep
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.perceptron.bias_saturation import clip_off_saturated_effective_bias
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.quantization_bounds import LAYER_WEIGHT_BITS
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner as Tuner,
)

BITS = 4
Q_MAX = 2 ** (BITS - 1) - 1  # 7
THETA = 1.25
PT = PerceptronTransformer()


def _bias_heavy(w_max: float = 0.12, b_max: float = 0.98) -> Perceptron:
    """max|w~| = 0.12 theta and max|b~| = 0.98 theta (theta-normalized effective units), a 4-bit T-LIF-like layer:
    s_w = floor(7 / 0.12) = 58 and r = ceil(0.98 * 58 / 7) = 9. Every channel can fire (no bias below the
    OFF-saturation bound the projection clips first)."""
    gen = torch.Generator().manual_seed(3)
    p = Perceptron(6, 10, normalization=nn.Identity())
    p.set_activation_scale(THETA)
    w = (torch.rand(6, 10, generator=gen) * 2 - 1) * w_max
    w[0, 0] = w_max
    w[:, 1] = w_max
    b = torch.tensor([0.31, b_max, 0.4137, -0.06, 0.77, -0.035])
    with torch.no_grad():
        p.layer.weight.copy_(w * THETA)
        p.layer.bias.copy_(b * THETA)
    return p


def _projected_bias_input(p) -> torch.Tensor:
    """The effective bias the projection codes: after its OFF-saturation clip (idempotent, function-preserving)."""
    ref = copy.deepcopy(p)
    clip_off_saturated_effective_bias(ref, transformer=PT)
    return PT.get_effective_bias(ref).detach().clone()


def _tuner_projection(*, register: bool, rate: float = 1.0, bits: int = BITS, float_bias: bool = False,
                      rows_floor=None):
    """The projection the WQ step's tuner applies (``_get_new_perceptron_transform``), with the route as the step
    names it. A tree whose tuner has no route reads its own projection here."""
    tuner = Tuner.__new__(Tuner)
    tuner.quantization_bits, tuner.two_scale_projection = bits, not float_bias
    tuner.fractional_scale, tuner.float_bias = False, float_bias
    tuner.bias_rows_floor = rows_floor
    tuner.register_bias = register
    tuner.pipeline = SimpleNamespace(config={"device": "cpu"})
    return tuner._get_new_perceptron_transform(rate)


def _codes(p) -> torch.Tensor:
    return PT.get_effective_bias(p) * float(p.parameter_scale)


# ---------------------------------------------------------------------------
# T-A1: the register route codes the bias on the weight lattice
# ---------------------------------------------------------------------------


def test_register_route_codes_every_bias_on_the_weight_lattice():
    """T-A1. Every bias code equals ``round(b~ s_w)`` and ``bias_scale == parameter_scale``. On R the codes are
    multiples of r = 9 (errors up to 4.5 / s_w)."""
    p = _bias_heavy()
    b_float = _projected_bias_input(p)
    _tuner_projection(register=True)(p)
    s_w = float(p.parameter_scale)
    assert s_w == 58.0
    assert float(p.bias_scale) == s_w, "the register route stamps the bias grid on the weight grid"
    codes = _codes(p)
    torch.testing.assert_close(codes, torch.round(codes), atol=1e-4, rtol=0)
    torch.testing.assert_close(torch.round(codes), torch.round(b_float * s_w), atol=0, rtol=0)
    assert float((PT.get_effective_bias(p) - b_float).abs().max()) <= 0.5 / s_w + 1e-6


def test_register_route_span_is_the_one_the_export_allocates():
    """The codes reach beyond the weight register's ±q_max (the register spans ±q_max r) and stay inside the span."""
    p = _bias_heavy()
    _tuner_projection(register=True)(p)
    s_w = float(p.parameter_scale)
    r = math.ceil(0.98 * s_w / Q_MAX)
    codes = torch.round(_codes(p))
    assert float(codes.abs().max()) > Q_MAX
    assert float(codes.max()) <= Q_MAX * r and float(codes.min()) >= -(Q_MAX + 1) * r


def test_register_route_weights_stay_on_the_weight_register():
    p = _bias_heavy()
    _tuner_projection(register=True)(p)
    w = PT.get_effective_weight(p) * float(p.parameter_scale)
    torch.testing.assert_close(w, torch.round(w), atol=1e-4, rtol=0)
    assert float(torch.round(w).abs().max()) <= Q_MAX


def test_register_route_rate_zero_is_the_identity_and_rates_interpolate():
    p0, ph = _bias_heavy(), _bias_heavy()
    raw_w, raw_b = p0.layer.weight.detach().clone(), p0.layer.bias.detach().clone()
    _tuner_projection(register=True, rate=0.0)(p0)
    assert torch.equal(p0.layer.weight.detach(), raw_w) and torch.equal(p0.layer.bias.detach(), raw_b)
    b_float = _projected_bias_input(ph)
    _tuner_projection(register=True, rate=0.5)(ph)
    s_w = float(ph.parameter_scale)
    expected = 0.5 * torch.round(b_float * s_w) / s_w + 0.5 * b_float
    torch.testing.assert_close(PT.get_effective_bias(ph), expected, atol=1e-6, rtol=0)


def test_a_core_type_of_its_own_width_keeps_the_route():
    """[WI-5g] a perceptron on a core type of another weight width projects on that width's grid, on the same route."""
    p = _bias_heavy()
    setattr(p, LAYER_WEIGHT_BITS, 5)
    b_float = _projected_bias_input(p)
    _tuner_projection(register=True)(p)
    s_w = float(p.parameter_scale)
    assert s_w == math.floor(15 / 0.12) and float(p.bias_scale) == s_w
    torch.testing.assert_close(torch.round(_codes(p)), torch.round(b_float * s_w), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# T-A4 (guard): the param-encoded route keeps its r-multiple grid byte for byte
# ---------------------------------------------------------------------------


def _reference_rows_grid(p, bits: int, floor=None) -> None:
    """R's two-scale projection (``_transform_two_scale`` at 3068ef59), transcribed op for op onto ``p``: the
    OFF-saturation clip, the weight grid ``floor(q_max / max|w|)``, the bias grid ``s_w / r`` with
    ``r = max(ceil(max|b| s_w / q_max), 1[, floor])``, both clamped to ±q_max on their own grid."""
    q_min, q_max = float(-(2 ** (bits - 1))), float(2 ** (bits - 1) - 1)

    def quantize(scale):
        return lambda param: torch.round(param * scale).clamp_(min=q_min, max=q_max) / scale

    with torch.no_grad():
        clip_off_saturated_effective_bias(p, transformer=PT)
        w, b = PT.get_effective_weight(p), PT.get_effective_bias(p)
        s_w = torch.clamp(torch.floor(q_max / torch.clamp(torch.max(torch.abs(w)), min=1e-12)), min=1.0)
        r = torch.clamp(torch.ceil(torch.max(torch.abs(b)) * s_w / q_max), min=1.0)
        if floor is not None:
            r = torch.clamp(r, min=float(floor))
        p.set_parameter_scale(s_w)
        p.set_bias_scale(s_w / r)
        PT.apply_effective_weight_transform(p, quantize(s_w))
        PT.apply_effective_bias_transform(p, quantize(s_w / r))


@pytest.mark.parametrize("floor", [None, 12])
def test_param_encoded_rows_route_is_byte_identical(floor):
    """T-A4 guard (green before and after): bias-row splitting keeps its equal-share r-multiple grid, byte for byte,
    through the tuner (the route not named) and through the projection itself."""
    ref, via_tuner, direct = _bias_heavy(), _bias_heavy(), _bias_heavy()
    _reference_rows_grid(ref, BITS, floor)
    _tuner_projection(register=False, rows_floor=floor)(via_tuner)
    NormalizationAwarePerceptronQuantization(BITS, "cpu", 1.0, two_scale=True, bias_rows_floor=floor).transform(direct)
    assert float(ref.parameter_scale) / float(ref.bias_scale) >= 2, "fixture: a coarser bias grid (r > 1)"
    for p in (via_tuner, direct):
        for (name, a), (_, b) in zip(p.named_parameters(), ref.named_parameters()):
            assert torch.equal(a.detach(), b.detach()), name

# ---------------------------------------------------------------------------
# The WQ step names the route
# ---------------------------------------------------------------------------


class _Pipeline:
    def __init__(self, config, entries):
        self.config = config
        self.entries = dict(entries)
        self.reporter = SimpleNamespace(report=lambda *a, **k: None)

    def get_entry(self, _step, key):
        return self.entries[key]

    def add_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value

    def update_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value


def _flow():
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4)).eval()
    return convert_torch_model(model, (8,), 4, device="cpu").eval()


def _captured(monkeypatch, **config) -> dict:
    captured: dict = {}
    monkeypatch.setattr(WeightQuantizationStep, "run_tuner", lambda self, cls, *a, **kw: captured.update(kw))
    monkeypatch.setattr(WeightQuantizationStep, "_canonicalize_starved_bias_outliers", lambda self, model: None)
    base = {"device": "cpu", "weight_quantization": True, "weight_bits": BITS, "wq_two_scale_projection": True}
    WeightQuantizationStep(_Pipeline({**base, **config}, {"model": _flow(), "adaptation_manager": object()})).process()
    return captured


def test_the_step_names_the_register_route_on_an_on_chip_bias_platform(monkeypatch):
    captured = _captured(monkeypatch, has_bias=True)
    assert captured["two_scale_projection"] is True
    assert captured.get("register_bias") is True


def test_the_step_keeps_the_rows_route_on_a_param_encoded_platform(monkeypatch):
    """Guard: bias-row splitting on a param-encoded platform is the rows route, never the register route."""
    captured = _captured(monkeypatch, has_bias=False, bias_row_splitting="auto")
    assert captured["two_scale_projection"] is True
    assert captured.get("register_bias", False) is False


def test_without_the_two_scale_flag_no_route_is_named(monkeypatch):
    captured = _captured(monkeypatch, has_bias=True, wq_two_scale_projection=False)
    assert captured["two_scale_projection"] is False
    assert captured.get("register_bias", False) is False
