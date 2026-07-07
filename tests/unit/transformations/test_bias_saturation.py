"""Saturation-equivalent effective-bias canonicalization (the WQ-entry crater fix).

The t01_19/t0_03 crater class: a dead (constant-OFF) channel carries a huge
negative effective bias (measured specimen: b_eff = -12.6 against w_max = 0.57,
via raw_bias +5.65 through a BN fold factor u = -4.14). NAPQ's shared
per-perceptron weight/bias scale (the chip contract) then sets its grid from
that functionally-unobservable scalar and rounds 100% of the layer's 4-bit
weights to zero — the model collapses to chance at the full-rate projection.

A channel with ``b + pos_reach <= 0`` outputs the activation floor for EVERY
input in [0,1], so any bias below ``-pos_reach`` is behaviorally equivalent to
``-pos_reach``. Clipping to that bound is function-preserving and keeps the
quantization scale honest. Measured on the specimen: float 0.9927 -> 0.9927
(bit-equal), NAPQ full-rate 0.1013 -> 0.9873.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.autograd import LeakyGradReLU
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.bias_saturation import (
    clip_off_saturated_effective_bias,
    off_saturation_bias_bound,
)
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)


def _make_perceptron(seed=0, normalization=None):
    torch.manual_seed(seed)
    p = Perceptron(4, 8, normalization=normalization or nn.Identity())
    p.activation = LeakyGradReLU()
    with torch.no_grad():
        p.layer.weight.data.mul_(0.3)
        if p.layer.bias is not None:
            p.layer.bias.data.mul_(0.3)
    return p


def _set_effective_bias_entry(p, channel, value):
    def setter(b):
        out = b.clone()
        out[channel] = value
        return out

    PerceptronTransformer().apply_effective_bias_transform(p, setter)


class TestOffSaturationBiasBound:
    def test_bound_is_negative_positive_reach_per_channel(self):
        w = torch.tensor([
            [1.0, -2.0, 0.5],
            [-1.0, -1.0, -1.0],
            [0.0, 0.25, 0.25],
        ])
        bound = off_saturation_bias_bound(w)
        assert torch.allclose(bound, torch.tensor([-1.5, 0.0, -0.5]))

    def test_conv_shaped_weights_flatten_per_output_channel(self):
        w = torch.zeros(2, 3, 2, 2)
        w[0] = 0.25  # pos_reach = 3*2*2*0.25 = 3.0
        w[1] = -1.0  # pos_reach = 0
        bound = off_saturation_bias_bound(w)
        assert torch.allclose(bound, torch.tensor([-3.0, 0.0]))


class TestClipOffSaturatedEffectiveBias:
    def test_off_channel_clips_to_the_bound(self):
        p = _make_perceptron(seed=0)
        _set_effective_bias_entry(p, 2, -20.0)
        pt = PerceptronTransformer()
        bound = off_saturation_bias_bound(pt.get_effective_weight(p))

        clipped = clip_off_saturated_effective_bias(p)

        # The random fixture may carry naturally-OFF channels too; the planted
        # outlier must land exactly on its bound.
        assert clipped >= 1
        b = pt.get_effective_bias(p)
        assert torch.isclose(b[2], bound[2], atol=1e-6)

    def test_clip_is_function_preserving_on_the_normalized_domain(self):
        torch.manual_seed(7)
        p = _make_perceptron(seed=1)
        _set_effective_bias_entry(p, 0, -15.0)
        x = torch.rand(64, 8)  # inputs in [0, 1]
        p.eval()
        with torch.no_grad():
            before = p(x).clone()
        clip_off_saturated_effective_bias(p)
        with torch.no_grad():
            after = p(x)
        assert torch.equal(before, after), (
            "clipping a constant-OFF channel must be bit-invisible in forward"
        )

    def test_clip_is_function_preserving_through_batchnorm(self):
        torch.manual_seed(11)
        p = _make_perceptron(seed=2, normalization=nn.BatchNorm1d(4))
        p.train()
        with torch.no_grad():
            p(torch.rand(128, 8))  # populate running stats
        p.eval()
        _set_effective_bias_entry(p, 1, -12.6)
        x = torch.rand(64, 8)
        with torch.no_grad():
            before = p(x).clone()
        clipped = clip_off_saturated_effective_bias(p)
        with torch.no_grad():
            after = p(x)
        assert clipped >= 1
        assert torch.allclose(before, after, atol=1e-6)

    def test_live_channels_are_untouched(self):
        p = _make_perceptron(seed=3)
        pt = PerceptronTransformer()
        before = pt.get_effective_bias(p).clone()
        _set_effective_bias_entry(p, 3, -30.0)

        clip_off_saturated_effective_bias(p)

        after = pt.get_effective_bias(p)
        for c in range(3):
            assert torch.isclose(after[c], before[c], atol=1e-6), c

    def test_idempotent(self):
        p = _make_perceptron(seed=4)
        _set_effective_bias_entry(p, 2, -20.0)
        assert clip_off_saturated_effective_bias(p) == 1
        w1 = p.layer.weight.data.clone()
        b1 = p.layer.bias.data.clone()
        assert clip_off_saturated_effective_bias(p) == 0
        assert torch.equal(p.layer.weight.data, w1)
        assert torch.equal(p.layer.bias.data, b1)

    def test_encoding_layer_is_skipped(self):
        p = _make_perceptron(seed=5)
        p.is_encoding_layer = True
        _set_effective_bias_entry(p, 2, -20.0)
        assert clip_off_saturated_effective_bias(p) == 0
        b = PerceptronTransformer().get_effective_bias(p)
        assert torch.isclose(b[2], torch.tensor(-20.0), atol=1e-4)

    def test_biasless_layer_is_a_noop(self):
        torch.manual_seed(6)
        p = Perceptron(4, 8, normalization=nn.Identity())
        p.layer.bias = None
        assert clip_off_saturated_effective_bias(p) == 0


class TestNapqScaleIsSaturationAware:
    """The crater regression: NAPQ's shared scale must derive from the
    canonicalized bias, never from unobservable dead-channel bias mass."""

    def _crater_perceptron(self, seed=8):
        # The t01_19 anatomy at 4 bits: w_max ~ 0.57, one dead channel at
        # b_eff = -12.6 -> grid step 1.8 -> every weight rounds to zero.
        p = _make_perceptron(seed=seed)
        _set_effective_bias_entry(p, 2, -12.6)
        return p

    def test_full_rate_projection_keeps_live_weights(self):
        p = self._crater_perceptron()
        NormalizationAwarePerceptronQuantization(bits=4, device="cpu", rate=1.0).transform(p)
        eff_w = PerceptronTransformer().get_effective_weight(p)
        nonzero = (eff_w.abs() > 0).float().mean().item()
        assert nonzero > 0.5, (
            f"a dead-channel bias outlier must not zero the weight grid "
            f"(nonzero fraction {nonzero:.3f})"
        )

    def test_parameter_scale_reflects_the_clipped_p_max(self):
        p = self._crater_perceptron()
        pt = PerceptronTransformer()
        clipped_ref = self._crater_perceptron()
        clip_off_saturated_effective_bias(clipped_ref)
        w = pt.get_effective_weight(clipped_ref)
        b = pt.get_effective_bias(clipped_ref)
        p_max = max(float(w.abs().max()), float(b.abs().max()), 1e-12)

        NormalizationAwarePerceptronQuantization(bits=4, device="cpu", rate=1.0).transform(p)
        q_max = 2 ** (4 - 1) - 1
        assert float(p.parameter_scale) == pytest.approx(q_max / p_max, rel=1e-5)

    def test_projection_is_function_preserving_up_to_quant_error(self):
        torch.manual_seed(9)
        p = self._crater_perceptron(seed=9)
        x = torch.rand(64, 8)
        p.eval()
        with torch.no_grad():
            before = p(x).clone()
        NormalizationAwarePerceptronQuantization(bits=4, device="cpu", rate=1.0).transform(p)
        with torch.no_grad():
            after = p(x)
        # 4-bit quantization error is bounded by half a grid step per weight
        # once the grid is healthy; the pre-fix behavior (all weights zeroed)
        # collapses outputs to the bias constant instead.
        assert (after - before).abs().max() < 0.5
        assert after.abs().sum() > 0

    def test_half_rate_blend_also_rides_the_clipped_state(self):
        p = self._crater_perceptron(seed=10)
        NormalizationAwarePerceptronQuantization(bits=4, device="cpu", rate=0.5).transform(p)
        b = PerceptronTransformer().get_effective_bias(p)
        assert b.abs().max() < 12.0, "the outlier must be clipped at any rate"
