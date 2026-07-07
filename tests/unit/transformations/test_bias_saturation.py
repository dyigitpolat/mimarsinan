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


class TestEmpiricalBiasBounds:
    """Guarded canonicalization math: shrink |b| on EMPIRICALLY saturated
    channels only as far as the observed pre-activation range proves harmless
    (margin-guarded); never grow |b|, never flip its sign."""

    def test_empirically_on_channel_shrinks_to_observed_slack(self):
        from mimarsinan.transformations.bias_saturation import (
            empirical_bias_shift,
        )

        b = torch.tensor([21.0])
        v_min = torch.tensor([19.0])
        v_max = torch.tensor([21.4])
        delta = empirical_bias_shift(b, v_min, v_max, ceiling=1.0, margin=0.25)
        assert torch.allclose(delta, torch.tensor([-17.75]))
        assert float(v_min + delta) >= 1.0

    def test_empirically_off_channel_shrinks_toward_zero(self):
        from mimarsinan.transformations.bias_saturation import (
            empirical_bias_shift,
        )

        b = torch.tensor([-52.0])
        v_min = torch.tensor([-55.0])
        v_max = torch.tensor([-49.0])
        delta = empirical_bias_shift(b, v_min, v_max, ceiling=1.0, margin=0.25)
        assert torch.allclose(delta, torch.tensor([48.75]))
        assert float(v_max + delta) <= 0.0

    def test_unsaturated_channel_is_untouched(self):
        from mimarsinan.transformations.bias_saturation import (
            empirical_bias_shift,
        )

        b = torch.tensor([0.4])
        delta = empirical_bias_shift(
            b, torch.tensor([-0.5]), torch.tensor([0.9]), ceiling=1.0, margin=0.25,
        )
        assert torch.equal(delta, torch.zeros(1))

    def test_never_grows_the_bias(self):
        from mimarsinan.transformations.bias_saturation import (
            empirical_bias_shift,
        )

        # Slack smaller than the margin: the shift would grow |b|; skip.
        b = torch.tensor([2.0])
        delta = empirical_bias_shift(
            b, torch.tensor([1.1]), torch.tensor([2.5]), ceiling=1.0, margin=0.25,
        )
        assert torch.equal(delta, torch.zeros(1))

    def test_never_flips_the_sign(self):
        from mimarsinan.transformations.bias_saturation import (
            empirical_bias_shift,
        )

        # Observed slack exceeds |b|: clamp at zero rather than crossing.
        b = torch.tensor([0.5])
        delta = empirical_bias_shift(
            b, torch.tensor([19.0]), torch.tensor([20.0]), ceiling=1.0, margin=0.25,
        )
        assert torch.allclose(b + delta, torch.tensor([0.0]))


class TestGuardedCanonicalization:
    """The t01_16/rep3 residual: an outlier bias that is NOT provably
    saturated but IS empirically constant. The guarded pass shrinks it using
    the observed pre-activation range and VERIFIES decision agreement on the
    calibration batches, restoring the perceptron on any flip."""

    def _chain(self, seed=21):
        # Production perceptrons at WQ entry clamp at theta (the effective
        # ceiling); Hardtanh(0, 1) is the theta=1 stand-in the ON-side
        # canonicalization contract requires.
        torch.manual_seed(seed)
        p1 = _make_perceptron(seed=seed)
        p1.activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
        p2 = _make_perceptron(seed=seed + 1)
        p2.activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
        with torch.no_grad():
            p2.layer = nn.Linear(4, 4)
            p2.layer.weight.data = torch.randn(4, 4) * 0.3
            # The planted channel (2) is decision-relevant downstream, so a
            # saturation-breaking shift MUST flip decisions (the guard's food):
            # with ch2 at its saturated 1.0, class 0 wins (+3 > +0.5); with
            # ch2 broken to ~0, class 1 wins.
            p2.layer.weight.data[:, 2] = torch.tensor([3.0, -3.0, 1.0, -1.0])
            p2.layer.bias.data = torch.tensor([0.0, 0.5, 0.0, 0.0])

        class _Chain(nn.Module):
            def __init__(self):
                super().__init__()
                self.p1, self.p2 = p1, p2

            def forward(self, x):
                return self.p2(self.p1(x))

            def get_perceptrons(self):
                return [self.p1, self.p2]

        return _Chain().eval()

    def test_empirically_on_outlier_is_shrunk_and_verified(self):
        from mimarsinan.transformations.bias_saturation import (
            canonicalize_starved_bias_outliers,
        )

        model = self._chain()
        p1 = model.get_perceptrons()[0]
        _set_effective_bias_entry(p1, 2, 21.0)
        torch.manual_seed(3)
        batches = [torch.rand(64, 8) for _ in range(3)]
        with torch.no_grad():
            before = [model(x).argmax(-1) for x in batches]

        report = canonicalize_starved_bias_outliers(model, batches, bits=4)

        pt = PerceptronTransformer()
        b = pt.get_effective_bias(p1)
        assert float(b[2]) < 21.0
        assert report["clipped"] >= 1 and report["restored"] == 0
        with torch.no_grad():
            after = [model(x).argmax(-1) for x in batches]
        for x, y in zip(before, after):
            assert torch.equal(x, y)

    def test_grid_step_is_refined_by_an_order_of_magnitude(self):
        from mimarsinan.transformations.bias_saturation import (
            canonicalize_starved_bias_outliers,
        )

        model = self._chain(seed=33)
        p1 = model.get_perceptrons()[0]
        _set_effective_bias_entry(p1, 2, 21.0)
        batches = [torch.rand(64, 8) for _ in range(3)]
        canonicalize_starved_bias_outliers(model, batches, bits=4)
        pt = PerceptronTransformer()
        b = pt.get_effective_bias(p1).abs()
        # The mechanism's contract: the grid ceiling collapses from the
        # unobservable 21 to the observed saturation slack (~1.4 here).
        assert float(b.max()) < 2.0, float(b.max())

    def test_healthy_perceptrons_are_not_touched(self):
        from mimarsinan.transformations.bias_saturation import (
            canonicalize_starved_bias_outliers,
        )

        model = self._chain(seed=5)
        p1 = model.get_perceptrons()[0]
        w1 = p1.layer.weight.data.clone()
        b1 = p1.layer.bias.data.clone()
        report = canonicalize_starved_bias_outliers(
            model, [torch.rand(32, 8)], bits=4,
        )
        assert report["clipped"] == 0
        assert torch.equal(p1.layer.weight.data, w1)
        assert torch.equal(p1.layer.bias.data, b1)

    def _wide_chain(self, seed, downstream_col2):
        """The rep3/ch43 anatomy: ch2's drive is many small weights with a
        huge reach, swinging BOTH sides of the ceiling around a +21 bias —
        unreachable by rungs 1-2. ``downstream_col2`` decides whether its
        removal is decision-invariant."""
        torch.manual_seed(seed)
        p1 = Perceptron(4, 64, normalization=nn.Identity())
        p1.activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
        p2 = Perceptron(4, 4, normalization=nn.Identity())
        p2.activation = nn.Identity()
        with torch.no_grad():
            p1.layer.weight.data.mul_(0.3)
            p1.layer.bias.data.mul_(0.3)
            p1.layer.weight.data[2] = -0.7
            p2.layer.weight.data = torch.randn(4, 4) * 0.3
            p2.layer.weight.data[:, 2] = torch.as_tensor(downstream_col2)
            p2.layer.bias.data = torch.tensor([0.0, 0.5, 0.0, 0.0])
        _set_effective_bias_entry(p1, 2, 21.0)

        class _Chain(nn.Module):
            def __init__(self):
                super().__init__()
                self.p1, self.p2 = p1, p2

            def forward(self, x):
                return self.p2(self.p1(x))

            def get_perceptrons(self):
                return [self.p1, self.p2]

        return _Chain().eval()

    def test_wild_nuisance_channel_is_removed_when_decision_invariant(self):
        from mimarsinan.transformations.bias_saturation import (
            canonicalize_starved_bias_outliers,
        )

        model = self._wide_chain(seed=13, downstream_col2=[0.0, 0.0, 0.0, 0.0])
        p1 = model.get_perceptrons()[0]
        batches = [torch.rand(64, 64) for _ in range(3)]
        with torch.no_grad():
            before = [model(x).argmax(-1) for x in batches]

        report = canonicalize_starved_bias_outliers(model, batches, bits=4)

        assert report["removed"] >= 1
        pt = PerceptronTransformer()
        b = pt.get_effective_bias(p1)
        w = pt.get_effective_weight(p1)
        assert float(b[2]) == pytest.approx(0.0, abs=1e-6)
        assert float(w[2].abs().max()) == pytest.approx(0.0, abs=1e-6)
        with torch.no_grad():
            after = [model(x).argmax(-1) for x in batches]
        for x, y in zip(before, after):
            assert torch.equal(x, y)

    def test_decision_relevant_wild_channel_is_restored(self):
        from mimarsinan.transformations.bias_saturation import (
            canonicalize_starved_bias_outliers,
        )

        model = self._wide_chain(seed=17, downstream_col2=[3.0, -3.0, 1.0, -1.0])
        p1 = model.get_perceptrons()[0]
        b_before = p1.layer.bias.data.clone()
        w_before = p1.layer.weight.data.clone()
        batches = [torch.rand(64, 64) for _ in range(3)]

        report = canonicalize_starved_bias_outliers(model, batches, bits=4)

        # Other channels may be legitimately rung-2-canonicalized; the WILD
        # decision-relevant channel itself must be restored bit-exactly.
        assert report["removal_restored"] >= 1
        assert torch.allclose(p1.layer.bias.data[2], b_before[2])
        assert torch.allclose(p1.layer.weight.data[2], w_before[2])

    def test_decision_flip_restores_the_perceptron(self, monkeypatch):
        from mimarsinan.transformations import bias_saturation

        model = self._chain(seed=8)
        p1 = model.get_perceptrons()[0]
        _set_effective_bias_entry(p1, 2, 21.0)
        b_before = p1.layer.bias.data.clone()
        # Force a saturation-breaking shift so the guard must fire.
        monkeypatch.setattr(
            bias_saturation, "empirical_bias_shift",
            lambda b, vmin, vmax, *, ceiling, margin: torch.where(
                b.abs() > 10.0, -b, torch.zeros_like(b),
            ),
        )
        report = bias_saturation.canonicalize_starved_bias_outliers(
            model, [torch.rand(64, 8)], bits=4,
        )
        assert report["restored"] >= 1
        assert torch.allclose(p1.layer.bias.data, b_before)


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
