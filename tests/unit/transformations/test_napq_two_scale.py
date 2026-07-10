"""Tests for the M2 two-scale ``NormalizationAwarePerceptronQuantization`` projection.

The shipped shared grid takes ONE scale per perceptron over weights AND bias
(``q_max / max(|w|, |b|)``); on the real cascaded-TTFS fc perceptrons the grid
is set by the bias (2-8x the largest weight) and 58-95% of weights round to
exactly zero, cratering the first-crossing forward
(``docs/research/findings/wq_cascade_crater_repair.md`` §3).

The two-scale repair derives the weight grid from ``max|w|`` alone and
quantizes the effective bias on its OWN ``q_max`` register range.  The bias
grid is integer-ratio-snapped to the weight grid
(``scale_b = scale_w / r`` with integer ``r``), so the quantized bias stays an
EXACT integer in weight-grid units (``b * scale_w = r * bias_int``) — the
lattice both the chip export and the NF<->SCM parity contract consume.
"""

import pickle

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)

BITS = 5
Q_MAX = (2 ** (BITS - 1)) - 1  # 15
Q_MIN = -(2 ** (BITS - 1))


def _bias_dominant_perceptron():
    """Effective |b|max (0.75) is 12.5x the effective |w|max (0.06): the shared
    grid step is 0.05, so most weights land below the half-step and collapse."""
    p = Perceptron(4, 8, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    weights = torch.tensor([
        [0.060, -0.050, 0.020, -0.015, 0.010, -0.005, 0.018, -0.022],
        [-0.030, 0.012, -0.008, 0.024, -0.016, 0.007, -0.011, 0.019],
        [0.045, -0.021, 0.013, -0.009, 0.023, -0.017, 0.006, -0.014],
        [-0.026, 0.033, -0.019, 0.010, -0.006, 0.021, -0.012, 0.008],
    ])
    biases = torch.tensor([0.75, 0.50, 0.30, 0.10])
    with torch.no_grad():
        p.layer.weight.data.copy_(weights)
        p.layer.bias.data.copy_(biases)
    return p


def _weight_dominant_perceptron():
    p = Perceptron(3, 6, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    weights = torch.tensor([
        [0.60, -0.50, 0.20, -0.15, 0.10, -0.05],
        [-0.30, 0.12, -0.08, 0.24, -0.16, 0.07],
        [0.45, -0.21, 0.13, -0.09, 0.23, -0.17],
    ])
    biases = torch.tensor([0.35, -0.20, 0.10])
    with torch.no_grad():
        p.layer.weight.data.copy_(weights)
        p.layer.bias.data.copy_(biases)
    return p


def _reference_shared_grid(param, scale):
    """The pre-M2 shared-grid projection (round, clamp, dequantize)."""
    return (param * scale).round().clamp(Q_MIN, Q_MAX) / scale


class TestTwoScaleRepairsWeightStarvation:
    def test_shared_grid_collapses_bias_dominated_weights(self):
        """Pathology reproduction: the shared grid zeroes most weights."""
        p = _bias_dominant_perceptron()
        float_w = PerceptronTransformer().get_effective_weight(p).clone()
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0
        ).transform(p)
        quant_w = PerceptronTransformer().get_effective_weight(p)

        zero_frac = float((quant_w == 0).float().mean())
        assert zero_frac >= 0.5, (
            f"expected the shared bias-set grid to annihilate most weights, "
            f"got zero fraction {zero_frac} (float max |w| = {float_w.abs().max()})"
        )

    def test_two_scale_weights_do_not_collapse(self):
        p = _bias_dominant_perceptron()
        float_w = PerceptronTransformer().get_effective_weight(p).clone()
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0, two_scale=True
        ).transform(p)
        quant_w = PerceptronTransformer().get_effective_weight(p)

        grid_step = float(float_w.abs().max()) / Q_MAX
        # Every weight is within half a WEIGHT-grid step of float, and only
        # weights genuinely below the half-step may round to zero.
        assert torch.all((quant_w - float_w).abs() <= grid_step / 2 + 1e-6), (
            f"max |dq - float| = {(quant_w - float_w).abs().max()}"
        )
        collapsed = quant_w == 0
        collapsible = float_w.abs() < grid_step / 2 + 1e-9
        assert torch.equal(collapsed, collapsible), (
            "weights above half a weight-grid step must survive the projection"
        )

    def test_two_scale_bias_matches_float_within_its_grid(self):
        p = _bias_dominant_perceptron()
        float_b = PerceptronTransformer().get_effective_bias(p).clone()
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0, two_scale=True
        ).transform(p)
        quant_b = PerceptronTransformer().get_effective_bias(p)
        bias_step = 1.0 / float(p.bias_scale)
        assert torch.all((quant_b - float_b).abs() <= bias_step / 2 + 1e-6)

    def test_two_scale_dequantized_product_matches_float(self):
        p = _bias_dominant_perceptron()
        pt = PerceptronTransformer()
        float_w = pt.get_effective_weight(p).clone()
        float_b = pt.get_effective_bias(p).clone()
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0, two_scale=True
        ).transform(p)
        quant_w = pt.get_effective_weight(p)
        quant_b = pt.get_effective_bias(p)

        x = torch.linspace(0.0, 1.0, steps=float_w.shape[1])
        y_float = float_w @ x + float_b
        y_quant = quant_w @ x + quant_b
        w_step = 1.0 / float(p.parameter_scale)
        b_step = 1.0 / float(p.bias_scale)
        tol = float(x.abs().sum()) * w_step / 2 + b_step / 2
        assert torch.all((y_quant - y_float).abs() <= tol), (
            f"max |y_quant - y_float| = {(y_quant - y_float).abs().max()} > {tol}"
        )


class TestTwoScaleGridContract:
    def test_scales_stamped_from_weights_and_bias_alone(self):
        p = _bias_dominant_perceptron()
        pt = PerceptronTransformer()
        w_max = float(pt.get_effective_weight(p).abs().max())
        b_max = float(pt.get_effective_bias(p).abs().max())
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0, two_scale=True
        ).transform(p)

        scale_w = float(p.parameter_scale)
        scale_b = float(p.bias_scale)
        assert scale_w == pytest.approx(Q_MAX / w_max, rel=1e-6), (
            "parameter_scale must derive from max|w| alone"
        )
        ratio = scale_w / scale_b
        assert ratio == pytest.approx(round(ratio), abs=1e-6), (
            f"bias grid must be integer-ratio-snapped to the weight grid, got {ratio}"
        )
        assert round(ratio) >= 2, "a bias-dominant perceptron needs a coarser bias grid"
        # The snapped bias grid STEP is within one weight-grid step of the
        # memo's pure q_max/max|b| grid (never finer than the register admits).
        natural_step = b_max / Q_MAX
        assert 1.0 / scale_b >= natural_step - 1e-9
        assert 1.0 / scale_b <= natural_step + 1.0 / scale_w + 1e-9

    def test_bias_integers_on_both_lattices(self):
        """The bias must be integer on its OWN grid (the q_max register range)
        AND on the weight grid (what the chip export emits as r * bias_int)."""
        p = _bias_dominant_perceptron()
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0, two_scale=True
        ).transform(p)
        quant_b = PerceptronTransformer().get_effective_bias(p)

        own = quant_b * p.bias_scale
        assert torch.allclose(own, torch.round(own), atol=1e-3, rtol=1e-3)
        assert float(torch.round(own).abs().max()) <= Q_MAX

        weight_lattice = quant_b * p.parameter_scale
        assert torch.allclose(
            weight_lattice, torch.round(weight_lattice), atol=1e-3, rtol=1e-3
        ), "quantized bias must stay exactly on the weight-grid integer lattice"

    def test_weight_dominant_ratio_is_one_and_matches_shared_grid(self):
        """When weights set the grid the two projections coincide exactly."""
        p_shared = _weight_dominant_perceptron()
        p_two = _weight_dominant_perceptron()
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0
        ).transform(p_shared)
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0, two_scale=True
        ).transform(p_two)

        assert float(p_two.parameter_scale) == float(p_shared.parameter_scale)
        assert float(p_two.bias_scale) == float(p_two.parameter_scale)
        assert torch.equal(p_two.layer.weight.data, p_shared.layer.weight.data)
        assert torch.equal(p_two.layer.bias.data, p_shared.layer.bias.data)

    def test_two_scale_rate_zero_is_identity(self):
        p = _bias_dominant_perceptron()
        orig_w = p.layer.weight.data.clone()
        orig_b = p.layer.bias.data.clone()
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=0.0, two_scale=True
        ).transform(p)
        assert torch.allclose(p.layer.weight.data, orig_w, atol=1e-6)
        assert torch.allclose(p.layer.bias.data, orig_b, atol=1e-6)

    def test_two_scale_without_bias_falls_back_to_weight_grid(self):
        p = Perceptron(4, 8, bias=False, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        with torch.no_grad():
            p.layer.weight.data.mul_(0.3)
        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0, two_scale=True
        ).transform(p)
        assert float(p.bias_scale) == float(p.parameter_scale)
        eff_w = PerceptronTransformer().get_effective_weight(p)
        scaled = eff_w * p.parameter_scale
        assert torch.allclose(scaled, torch.round(scaled), atol=1e-3)


class TestFlagOffIsByteIdentical:
    def test_shared_grid_output_matches_pre_change_algorithm(self):
        """two_scale=False must reproduce the shipped shared-grid projection
        bit-for-bit: one scale over max(|w|, |b|), both parameters on it."""
        p = _bias_dominant_perceptron()
        pt = PerceptronTransformer()
        float_w = pt.get_effective_weight(p).clone()
        float_b = pt.get_effective_bias(p).clone()
        p_max = torch.clamp(
            torch.maximum(float_w.abs().max(), float_b.abs().max()), min=1e-12
        )
        scale = Q_MAX * (1.0 / p_max)

        NormalizationAwarePerceptronQuantization(
            bits=BITS, device="cpu", rate=1.0
        ).transform(p)

        assert torch.equal(
            p.layer.weight.data, _reference_shared_grid(float_w, scale)
        ), "flag-off weights must be byte-identical to the shared-grid algorithm"
        assert torch.equal(
            p.layer.bias.data, _reference_shared_grid(float_b, scale)
        ), "flag-off bias must be byte-identical to the shared-grid algorithm"
        assert float(p.parameter_scale) == float(scale)
        assert float(p.bias_scale) == float(scale), (
            "flag-off must stamp bias_scale == parameter_scale (shared grid)"
        )

    def test_default_is_shared_grid(self):
        import inspect

        sig = inspect.signature(NormalizationAwarePerceptronQuantization.__init__)
        assert sig.parameters["two_scale"].default is False, (
            "two-scale must be opt-in (byte-identity of existing runs)"
        )


class TestPerceptronBiasScaleState:
    def test_fresh_perceptron_defaults_to_unit_bias_scale(self):
        p = Perceptron(2, 3, normalization=nn.Identity())
        assert float(p.bias_scale) == 1.0
        assert p.bias_scale.requires_grad is False

    def test_set_bias_scale_accepts_float_and_tensor(self):
        p = Perceptron(2, 3, normalization=nn.Identity())
        p.set_bias_scale(2.5)
        assert float(p.bias_scale) == 2.5
        p.set_bias_scale(torch.tensor(4.0))
        assert float(p.bias_scale) == 4.0

    def test_set_parameter_scale_re_declares_the_shared_grid(self):
        """Legacy single-scale callers must stay coherent: declaring a
        parameter_scale without a bias grid means the SHARED grid."""
        p = Perceptron(2, 3, normalization=nn.Identity())
        p.set_bias_scale(4.0)
        p.set_parameter_scale(20.0)
        assert float(p.bias_scale) == 20.0
        p.set_bias_scale(5.0)
        assert float(p.parameter_scale) == 20.0, (
            "refining the bias grid must not move the weight grid"
        )
        assert float(p.bias_scale) == 5.0

    def test_unpickling_pre_two_scale_cache_inherits_parameter_scale(self):
        """Old cached models carry no bias_scale; the shared-grid semantics of
        every pre-two-scale artifact is bias_scale == parameter_scale."""
        p = Perceptron(2, 3, normalization=nn.Identity())
        p.set_parameter_scale(20.0)
        del p._parameters["bias_scale"]
        restored = pickle.loads(pickle.dumps(p))
        assert float(restored.bias_scale) == 20.0
