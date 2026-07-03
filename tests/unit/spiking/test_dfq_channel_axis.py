"""W-CAL-1: DFQ per-channel statistics reduce over the perceptron's TRUE channel axis.

The channel axis is owner-declared ground truth (``Perceptron.output_channel_axis``):
``nn.Linear``-driven perceptrons put channels on the LAST dim (2-D MLP outputs,
token-major 3-D mixers); conv mappers drive the shared perceptron through
``F.conv`` and put channels on dim 1 ([B,C,H,W] / [B,C,L]). Reducing over the
last dim on conv activations wrote per-image-column gaps into channel biases
(the t0_18 entry crater); channels-last vehicles were CORRECT and must stay
bit-identical.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import (
    Perceptron,
    activation_channel_axis,
)
from mimarsinan.spiking.dfq_bias_correction import (
    channel_mean,
    dfq_correct_biases,
    mean_abs_gap,
    perceptron_channel_mean,
)


def _legacy_channel_mean(t):
    """The pre-fix reduction: mean over all dims except the LAST."""
    return t.reshape(-1, t.shape[-1]).float().mean(0)


# ── the channel-axis ground truth ──────────────────────────────────────────────


class TestActivationChannelAxis:
    def test_native_perceptron_declares_channels_last(self):
        p = Perceptron(6, 4)
        assert p.output_channel_axis == -1

    def test_2d_linear_output_resolves_to_last_dim(self):
        p = Perceptron(6, 4)
        t = torch.randn(8, 6)
        assert activation_channel_axis(p, t) == 1

    def test_3d_token_major_mixer_output_resolves_to_last_dim(self):
        p = Perceptron(6, 4)
        t = torch.randn(8, 5, 6)  # [B, tokens, C] — channels LAST
        assert activation_channel_axis(p, t) == 2

    def test_conv_marked_perceptron_resolves_to_dim_1(self):
        p = Perceptron(6, 4)
        p.output_channel_axis = 1  # what the conv mappers declare
        t = torch.randn(8, 6, 7, 7)  # [B, C, H, W]
        assert activation_channel_axis(p, t) == 1

    def test_missing_declaration_fails_loud(self):
        class Undeclared:
            layer = nn.Linear(4, 6)

        with pytest.raises(ValueError, match="output_channel_axis"):
            activation_channel_axis(Undeclared(), torch.randn(8, 6))

    def test_inconsistent_declaration_fails_loud(self):
        p = Perceptron(6, 4)
        # channels-last declared, but the last dim is NOT the channel count
        t = torch.randn(8, 6, 7, 7)
        with pytest.raises(ValueError, match="channel"):
            activation_channel_axis(p, t)

    def test_conv2d_mapper_declares_channels_first(self):
        from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper

        mapper = Conv2DPerceptronMapper(
            None, in_channels=2, out_channels=6, kernel_size=3, padding=1,
        )
        assert mapper.perceptron.output_channel_axis == 1

    def test_conv1d_mapper_declares_channels_first(self):
        from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper

        mapper = Conv1DPerceptronMapper(
            None, in_channels=2, out_channels=6, kernel_size=3, padding=1,
        )
        assert mapper.perceptron.output_channel_axis == 1

    def test_pre_declaration_pickled_perceptron_loads_channels_last(self):
        """Step caches saved before the layout declaration must load with the
        nn.Linear channels-last layout (the owner re-declares otherwise)."""
        import io

        p = Perceptron(6, 4)
        del p.__dict__["output_channel_axis"]  # simulate an old cache
        buf = io.BytesIO()
        torch.save(p, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)
        assert loaded.output_channel_axis == -1

    def test_pre_declaration_pickled_conv_mapper_restamps_channels_first(self):
        import io

        from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper

        mapper = Conv2DPerceptronMapper(
            None, in_channels=2, out_channels=6, kernel_size=3, padding=1,
        )
        del mapper.perceptron.__dict__["output_channel_axis"]
        buf = io.BytesIO()
        torch.save(mapper, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)
        assert loaded.perceptron.output_channel_axis == 1


# ── channel_mean on the resolved axis ──────────────────────────────────────────


class TestChannelMean:
    def test_channels_last_is_bit_identical_to_legacy_2d(self):
        t = torch.randn(32, 6)
        assert torch.equal(channel_mean(t, -1), _legacy_channel_mean(t))

    def test_channels_last_is_bit_identical_to_legacy_3d(self):
        t = torch.randn(8, 5, 6)
        assert torch.equal(channel_mean(t, 2), _legacy_channel_mean(t))

    def test_conv_activation_reduces_per_channel(self):
        t = torch.randn(8, 6, 4, 4)  # [B, C, H, W], C != W
        cm = channel_mean(t, 1)
        assert cm.shape == (6,)
        torch.testing.assert_close(cm, t.float().mean(dim=(0, 2, 3)))

    def test_old_last_dim_behavior_is_gone_for_conv(self):
        """Regression: dim-1 reduction of [B,C,H,W] is per-CHANNEL (len C), not
        the legacy per-image-column vector (len W)."""
        t = torch.randn(8, 6, 4, 4)
        p = Perceptron(6, 4)
        p.output_channel_axis = 1
        cm = perceptron_channel_mean(p, t)
        assert cm.shape == (6,)
        assert _legacy_channel_mean(t).shape == (4,)

    def test_perceptron_channel_mean_matches_declared_axis(self):
        p = Perceptron(6, 4)
        t = torch.randn(8, 5, 6)
        assert torch.equal(perceptron_channel_mean(p, t), _legacy_channel_mean(t))


# ── DFQ end-to-end on both layouts ─────────────────────────────────────────────


class _FakePerceptron:
    def __init__(self, out_features, *, axis=-1):
        self.layer = nn.Linear(out_features, out_features)
        self.layer.bias.data.zero_()
        self.output_channel_axis = axis


class _ConvFakeModel:
    """Cascade values are [B, C, H, W] with per-channel mean == bias + offset
    (H == W != C so the legacy last-dim reduction is shape-detectably wrong)."""

    def __init__(self, channels=6, spatial=4, offset=0.5):
        self._perceptrons = [_FakePerceptron(channels, axis=1)]
        self._spatial = spatial
        self._offset = offset

    def get_perceptrons(self):
        return self._perceptrons

    def cascade_means(self):
        out = {}
        for k, p in enumerate(self._perceptrons):
            per_channel = p.layer.bias.detach() + self._offset
            out[k] = per_channel.view(1, -1, 1, 1).expand(
                2, -1, self._spatial, self._spatial
            )
        return out


class _ChannelsLastFakeModel:
    """Token-major [B, T, C] cascade values with channel mean == bias + offset."""

    def __init__(self, channels=5, tokens=3, offset=0.5, n_perceptrons=2):
        self._perceptrons = [
            _FakePerceptron(channels) for _ in range(n_perceptrons)
        ]
        self._tokens = tokens
        self._offset = offset

    def get_perceptrons(self):
        return self._perceptrons

    def cascade_means(self):
        return {
            k: (p.layer.bias.detach() + self._offset)
            .view(1, 1, -1)
            .expand(2, self._tokens, -1)
            for k, p in enumerate(self._perceptrons)
        }


def _targets(model, values):
    return {
        k: torch.as_tensor(values, dtype=torch.float32)
        for k, _ in enumerate(model.get_perceptrons())
    }


def _legacy_dfq(model, ann_mean, cascade_means_fn, *, bias_iters, eta):
    """The pre-fix loop, verbatim semantics (last-dim reduction, min-truncation)."""
    perceptrons = list(model.get_perceptrons())
    for _ in range(bias_iters):
        cascade = cascade_means_fn()
        for k, perceptron in enumerate(perceptrons):
            cascade_value = cascade.get(k)
            bias = getattr(perceptron.layer, "bias", None)
            if cascade_value is None or k not in ann_mean or bias is None:
                continue
            cm = _legacy_channel_mean(cascade_value)
            ann_mu = ann_mean[k]
            n = min(cm.numel(), ann_mu.numel(), bias.numel())
            with torch.no_grad():
                bias[:n] += eta * (ann_mu[:n] - cm[:n]).to(bias.device, bias.dtype)


class TestConvDfqConvergence:
    def test_converges_to_injected_per_channel_gap(self):
        target = [1.0, -0.5, 0.25, 2.0, 0.0, -1.5]
        model = _ConvFakeModel(channels=6, spatial=4, offset=0.5)
        ann_mean = _targets(model, target)

        dfq_correct_biases(
            model, ann_mean, model.cascade_means, bias_iters=30, eta=0.7,
        )
        # fixed point: cascade channel-mean = bias + 0.5 = target
        expected = torch.tensor(target) - 0.5
        torch.testing.assert_close(
            model.get_perceptrons()[0].layer.bias.detach(), expected,
            rtol=0, atol=1e-3,
        )

    def test_legacy_axis_left_channels_beyond_width_untouched(self):
        """Documents the bug being fixed: the legacy loop only ever wrote the
        first min(W, C) rows; the fixed loop writes all C rows."""
        target = [1.0, -0.5, 0.25, 2.0, 0.0, -1.5]
        legacy = _ConvFakeModel(channels=6, spatial=4, offset=0.5)
        _legacy_dfq(
            legacy, _targets(legacy, target), legacy.cascade_means,
            bias_iters=30, eta=0.7,
        )
        legacy_bias = legacy.get_perceptrons()[0].layer.bias.detach()
        assert torch.equal(legacy_bias[4:], torch.zeros(2)), (
            "fixture must reproduce the legacy truncation for this to be a "
            "meaningful regression test"
        )

        fixed = _ConvFakeModel(channels=6, spatial=4, offset=0.5)
        dfq_correct_biases(
            fixed, _targets(fixed, target), fixed.cascade_means,
            bias_iters=30, eta=0.7,
        )
        fixed_bias = fixed.get_perceptrons()[0].layer.bias.detach()
        assert not torch.equal(fixed_bias[4:], torch.zeros(2))

    def test_gap_metric_measures_the_true_per_channel_gap(self):
        """The self-concealing metric is gone: with cascade = target + 0.3 per
        channel, mean_abs_gap reads exactly 0.3."""
        model = _ConvFakeModel(channels=6, spatial=4, offset=0.3)
        model.get_perceptrons()[0].layer.bias.data.copy_(
            torch.tensor([1.0, -0.5, 0.25, 2.0, 0.0, -1.5])
        )
        ann_mean = {
            0: model.get_perceptrons()[0].layer.bias.detach().clone()
        }
        gap = mean_abs_gap(
            model.get_perceptrons(), ann_mean, model.cascade_means(),
        )
        assert gap == pytest.approx(0.3, abs=1e-6)


class TestChannelsLastBitIdentity:
    """The CRITICAL regression guard: channels-last vehicles (mixers, 2-D MLP
    outputs) were correct before the fix — their DFQ output is bit-identical."""

    def test_dfq_biases_bit_identical_to_legacy_on_channels_last(self):
        target = [0.8, -0.2, 1.1, 0.0, -0.7]
        torch.manual_seed(0)
        legacy = _ChannelsLastFakeModel()
        _legacy_dfq(
            legacy, _targets(legacy, target), legacy.cascade_means,
            bias_iters=15, eta=0.7,
        )
        torch.manual_seed(0)
        fixed = _ChannelsLastFakeModel()
        dfq_correct_biases(
            fixed, _targets(fixed, target), fixed.cascade_means,
            bias_iters=15, eta=0.7,
        )
        for pl, pf in zip(legacy.get_perceptrons(), fixed.get_perceptrons()):
            assert torch.equal(pl.layer.bias.detach(), pf.layer.bias.detach())

    def test_mean_abs_gap_bit_identical_to_legacy_on_channels_last(self):
        model = _ChannelsLastFakeModel()
        ann_mean = _targets(model, [0.8, -0.2, 1.1, 0.0, -0.7])
        cascade = model.cascade_means()

        legacy_gaps = []
        for k, ann_mu in ann_mean.items():
            cm = _legacy_channel_mean(cascade[k])
            n = min(cm.numel(), ann_mu.numel())
            legacy_gaps.append((cm[:n] - ann_mu[:n]).abs().mean().item())
        legacy_gap = sum(legacy_gaps) / len(legacy_gaps)

        assert mean_abs_gap(
            model.get_perceptrons(), ann_mean, cascade,
        ) == legacy_gap
