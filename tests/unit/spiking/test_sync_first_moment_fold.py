"""[S3/R6] sequential first-moment fold: own-offset-EXCLUDED, sequential, mask-aware.

The sign trap (sync_deployment_exactness.md §3.2) is encoded here as an
executable fact: folding the RAW deployed-vs-float pre-activation mean gap
cancels the +theta/(2S) mid-tread half-step it rides on (0.93->0.59 measured);
the correct closed form excludes each hop's own intentional offset and folds
only the propagated upstream error.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support.bias_compensation import (
    apply_sync_exact_entry_half_step,
)
from mimarsinan.models.nn.decorators.clamp_quantize import TTFSCeilStaircaseDecorator
from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.dfq_bias_correction import (
    preactivation_channel_means,
    perceptron_preactivation_samples,
    sequential_first_moment_fold,
)
from mimarsinan.spiking.sync_first_moment import (
    SYNC_FIRST_MOMENT_FLAG,
    apply_sync_first_moment_fold,
    perceptron_forward_order,
    sync_half_step_own_offsets,
)

S = 4
HALF = 1.0 / (2 * S)  # theta = 1 everywhere below


def _perceptron(seed, in_f=4, out_f=4):
    torch.manual_seed(seed)
    p = Perceptron(out_f, in_f, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    with torch.no_grad():
        # Positive gains keep pre-activations inside (0, 1) and propagate
        # upstream staircase drift same-sign (the regime the memo describes).
        p.layer.weight.data.uniform_(0.05, 0.30)
        p.layer.bias.data.uniform_(0.05, 0.15)
    return p


class _Chain(nn.Module):
    def __init__(self, hops):
        super().__init__()
        self.hops = nn.ModuleList(hops)

    def forward(self, x):
        out = x
        for p in self.hops:
            out = p(out)
        return out

    def get_perceptrons(self):
        return list(self.hops)


def _install_staircase(model, steps=S):
    for p in model.get_perceptrons():
        p.activation = TransformedActivation(
            nn.Identity(),
            [TTFSCeilStaircaseDecorator(steps, p.activation_scale)],
        )


def _cal_x(n=2048, dim=4):
    torch.manual_seed(99)
    return torch.rand(n, dim)


def _biases(model):
    return [p.layer.bias.detach().clone() for p in model.get_perceptrons()]


class TestPreactivationCapture:
    def test_captures_the_activation_input(self):
        p = _perceptron(0)
        model = _Chain([p])
        x = _cal_x(16)
        samples = perceptron_preactivation_samples(model, x)
        expected = x @ p.layer.weight.detach().T + p.layer.bias.detach()
        torch.testing.assert_close(samples[0], expected)

    def test_channel_means_match_manual_means(self):
        p = _perceptron(1)
        model = _Chain([p])
        x = _cal_x(64)
        means = preactivation_channel_means(model, x)
        expected = (x @ p.layer.weight.detach().T + p.layer.bias.detach()).mean(0)
        torch.testing.assert_close(means[0], expected)


class TestOwnOffsetExclusionSignTrap:
    """The load-bearing exclusion: with the half-step baked and zero upstream
    error, the correct fold is a NO-OP; the naive fold cancels the half-step."""

    def _deployed_chain(self):
        from mimarsinan.spiking.dfq_bias_correction import teacher_channel_means

        model = _Chain([_perceptron(2), _perceptron(3)])
        x = _cal_x()
        float_ref = preactivation_channel_means(model, x)
        float_out = teacher_channel_means(model, x)
        folded = apply_sync_exact_entry_half_step(
            model, S, encoding_layer_placement="subsume",
        )
        assert folded == 2
        _install_staircase(model)
        return model, x, float_ref, float_out

    def test_correct_fold_preserves_the_half_step(self):
        model, x, float_ref, _ = self._deployed_chain()
        before = _biases(model)
        stats = apply_sync_first_moment_fold(model, x, float_ref, S)
        assert stats["folded"] == 2
        # Hop 1 has zero upstream error: gap == own offset exactly, delta ~ 0.
        torch.testing.assert_close(
            model.get_perceptrons()[0].layer.bias.detach(), before[0],
            rtol=0, atol=1e-5,
        )
        # Post-fold, every hop's residual gap equals its own offset exactly
        # (the fold is linear in the bias): the mid-tread compensation stands.
        after_gap = preactivation_channel_means(model, x)
        for k, mu in float_ref.items():
            torch.testing.assert_close(
                after_gap[k] - mu,
                torch.full_like(mu, HALF),
                rtol=0, atol=1e-5,
            )

    def test_naive_fold_reproduces_the_trap(self):
        model, x, float_ref, _ = self._deployed_chain()
        float_bias0 = (
            model.get_perceptrons()[0].layer.bias.detach() - HALF
        )
        sequential_first_moment_fold(
            model, float_ref, x, own_offsets={},  # the violation
        )
        # The half-step is cancelled: hop-1 bias returns to its FLOAT value.
        torch.testing.assert_close(
            model.get_perceptrons()[0].layer.bias.detach(), float_bias0,
            rtol=0, atol=1e-5,
        )

    def test_naive_fold_restores_the_floor_bias_on_outputs(self):
        model, x, float_ref, float_out = self._deployed_chain()

        def hop1_output_mean_error(m):
            p1 = m.get_perceptrons()[0]
            with torch.no_grad():
                a1 = p1(x)
            return float((a1.mean(0) - float_out[0]).mean())

        naive = _Chain([_perceptron(2), _perceptron(3)])
        apply_sync_exact_entry_half_step(
            naive, S, encoding_layer_placement="subsume",
        )
        _install_staircase(naive)
        sequential_first_moment_fold(naive, float_ref, x, own_offsets={})

        correct = model
        apply_sync_first_moment_fold(correct, x, float_ref, S)

        err_correct = hop1_output_mean_error(correct)
        err_naive = hop1_output_mean_error(naive)
        # Mid-tread intact: near-centered. Cancelled: the -1/(2S)-scale floor
        # drift returns (same-sign, whole-population).
        assert abs(err_correct) < 0.04
        assert err_naive < err_correct - 0.05


class TestSequentialOrdering:
    """Each hop's estimate is measured AFTER upstream folds landed: a repaired
    upstream systematic must not be double-folded into downstream biases."""

    def test_downstream_hop_sees_the_corrected_prefix(self):
        model = _Chain([_perceptron(4), _perceptron(5)])
        x = _cal_x()
        float_ref = preactivation_channel_means(model, x)
        # Half-step baked: the staircase is near-centered, so hop 2's honest
        # residual fold is small and the counterfactual isolates the injected
        # systematic below.
        apply_sync_exact_entry_half_step(
            model, S, encoding_layer_placement="subsume",
        )
        _install_staircase(model)

        p1, p2 = model.get_perceptrons()
        b1_deployed = p1.layer.bias.detach().clone()  # float + half-step
        b2_before = p2.layer.bias.detach().clone()
        with torch.no_grad():
            p1.layer.bias.data += 0.3  # an injected upstream systematic

        # Counterfactual: what a NON-sequential fold would push into hop 2 —
        # its raw pre-fold gap minus its own half-step offset.
        pre_fold_delta2 = (
            preactivation_channel_means(model, x)[1] - float_ref[1] - HALF
        )
        assert float(pre_fold_delta2.abs().mean()) > 0.05, "systematic must bite"

        apply_sync_first_moment_fold(model, x, float_ref, S)

        # Hop 1 was restored first (the 0.3 folded out, half-step kept)...
        torch.testing.assert_close(
            p1.layer.bias.detach(), b1_deployed, rtol=0, atol=1e-4,
        )
        # ...so hop 2 folded only the residual staircase drift, NOT the
        # already-repaired 0.3 systematic.
        b2_change = (p2.layer.bias.detach() - b2_before).abs().mean()
        assert float(b2_change) < 0.3 * float(pre_fold_delta2.abs().mean())


class TestMaskAndEdgeBehavior:
    def test_structurally_dead_rows_receive_no_fold(self):
        model = _Chain([_perceptron(6), _perceptron(7)])
        x = _cal_x()
        float_ref = preactivation_channel_means(model, x)
        _install_staircase(model)
        p2 = model.get_perceptrons()[1]
        mask = torch.zeros(4, dtype=torch.bool)
        mask[0] = True
        p2.layer.prune_bias_mask = mask
        with torch.no_grad():
            p2.layer.bias.data[0] = 0.0
        apply_sync_first_moment_fold(model, x, float_ref, S)
        assert float(p2.layer.bias.detach()[0]) == 0.0

    def test_fold_is_idempotent_per_perceptron(self):
        model = _Chain([_perceptron(8)])
        x = _cal_x()
        float_ref = preactivation_channel_means(model, x)
        _install_staircase(model)
        first = apply_sync_first_moment_fold(model, x, float_ref, S)
        assert first["folded"] == 1
        assert getattr(model.get_perceptrons()[0], SYNC_FIRST_MOMENT_FLAG)
        before = _biases(model)
        second = apply_sync_first_moment_fold(model, x, float_ref, S)
        assert second["folded"] == 0
        for b, p in zip(before, model.get_perceptrons()):
            assert torch.equal(p.layer.bias.detach(), b)

    def test_bias_free_hop_is_skipped(self):
        p = Perceptron(4, 4, bias=False, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        model = _Chain([p])
        x = _cal_x()
        float_ref = preactivation_channel_means(model, x)
        _install_staircase(model)
        stats = apply_sync_first_moment_fold(model, x, float_ref, S)
        assert stats["folded"] == 0
        assert stats["skipped"] == 1

    def test_missing_float_reference_is_skipped(self):
        model = _Chain([_perceptron(9)])
        x = _cal_x()
        _install_staircase(model)
        before = _biases(model)
        stats = apply_sync_first_moment_fold(model, x, {}, S)
        assert stats["folded"] == 0
        assert torch.equal(model.get_perceptrons()[0].layer.bias.detach(), before[0])


class TestSyncWrapperDerivations:
    def test_own_offsets_follow_the_baked_flag(self):
        flagged = _perceptron(10)
        plain = _perceptron(11)
        model = _Chain([flagged, plain])
        apply_sync_exact_entry_half_step(
            _Chain([flagged]), S, encoding_layer_placement="subsume",
        )
        offsets = sync_half_step_own_offsets(model, S)
        assert set(offsets) == {0}
        assert float(offsets[0]) == pytest.approx(HALF)

    def test_offsets_scale_with_theta_and_s(self):
        p = _perceptron(12)
        p.set_activation_scale(2.5)
        apply_sync_exact_entry_half_step(
            _Chain([p]), 8, encoding_layer_placement="subsume",
        )
        offsets = sync_half_step_own_offsets(_Chain([p]), 8)
        assert float(offsets[0]) == pytest.approx(2.5 / 16.0)

    def test_forward_order_defaults_to_declaration_order(self):
        model = _Chain([_perceptron(13), _perceptron(14)])
        assert perceptron_forward_order(model) == [0, 1]
