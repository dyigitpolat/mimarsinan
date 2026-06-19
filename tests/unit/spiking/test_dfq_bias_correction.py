"""Shared DFQ per-neuron bias-correction core (used by TTFS and LIF distmatch).

``dfq_correct_biases`` runs a few rounds of ``layer.bias += eta * (ann_mean -
cascade_mean)`` over a model's perceptrons, where the per-perceptron cascade
channel-mean is supplied by an injected ``cascade_means_fn`` (TTFS single-spike
cascade / LIF cycle-accurate cascade). It is mode-agnostic: the loop, the gap
measurement, and the teacher-mean capture live here; only the cascade readout
differs between modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.spiking.dfq_bias_correction import (
    channel_mean,
    dfq_correct_biases,
    mean_abs_gap,
)


class _FakePerceptron:
    def __init__(self, out_features):
        self.layer = nn.Linear(out_features, out_features)
        self.layer.bias.data.zero_()


class _FakeModel:
    """A model whose decoded cascade mean for perceptron k is exactly its bias
    plus a fixed offset — so the DFQ fixed point is bias = target - offset."""

    def __init__(self, out_features, n_perceptrons, offset=0.5):
        self._perceptrons = [_FakePerceptron(out_features) for _ in range(n_perceptrons)]
        self._offset = offset

    def get_perceptrons(self):
        return self._perceptrons

    def cascade_means(self):
        return {
            k: p.layer.bias.detach() + self._offset
            for k, p in enumerate(self._perceptrons)
        }


def _targets(model, value=1.0):
    return {
        k: torch.full_like(p.layer.bias.detach(), value)
        for k, p in enumerate(model.get_perceptrons())
    }


class TestChannelMean:
    def test_mean_over_all_but_last_dim(self):
        t = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        cm = channel_mean(t)
        assert cm.shape == (4,)
        torch.testing.assert_close(cm, t.reshape(-1, 4).mean(0))


class TestDfqShrinksGap:
    def test_gap_drops_toward_zero(self):
        torch.manual_seed(0)
        model = _FakeModel(out_features=4, n_perceptrons=3, offset=0.5)
        ann_mean = _targets(model, value=1.0)

        gap_before = mean_abs_gap(ann_mean, model.cascade_means())
        stats = dfq_correct_biases(
            model, ann_mean, model.cascade_means, bias_iters=15, eta=0.7,
        )
        gap_after = mean_abs_gap(ann_mean, model.cascade_means())

        assert gap_after < gap_before
        assert stats["mean_gap_after"] < stats["mean_gap_before"]
        assert stats["mean_gap_after"] == gap_after
        assert gap_after < 1e-3, "the simple fixed point must converge"

    def test_biases_move_toward_fixed_point(self):
        model = _FakeModel(out_features=2, n_perceptrons=2, offset=0.5)
        ann_mean = _targets(model, value=1.0)
        dfq_correct_biases(model, ann_mean, model.cascade_means, bias_iters=30, eta=0.7)
        for p in model.get_perceptrons():
            # fixed point: cascade = bias + 0.5 = 1.0  ->  bias = 0.5
            torch.testing.assert_close(
                p.layer.bias.detach(), torch.full_like(p.layer.bias.detach(), 0.5),
                rtol=0, atol=1e-3,
            )


class TestZeroItersIsNoOp:
    def test_zero_bias_iters_leaves_biases_unchanged(self):
        model = _FakeModel(out_features=3, n_perceptrons=2)
        ann_mean = _targets(model, value=1.0)
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        dfq_correct_biases(model, ann_mean, model.cascade_means, bias_iters=0, eta=0.7)
        for b, p in zip(before, model.get_perceptrons()):
            torch.testing.assert_close(p.layer.bias.detach(), b, rtol=0, atol=0)


class TestEtaScalesStep:
    def test_zero_eta_leaves_biases_unchanged(self):
        model = _FakeModel(out_features=3, n_perceptrons=2)
        ann_mean = _targets(model, value=1.0)
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        dfq_correct_biases(model, ann_mean, model.cascade_means, bias_iters=10, eta=0.0)
        for b, p in zip(before, model.get_perceptrons()):
            torch.testing.assert_close(p.layer.bias.detach(), b, rtol=0, atol=0)


class TestMissingEntriesSkipped:
    def test_perceptron_without_cascade_value_is_skipped(self):
        model = _FakeModel(out_features=2, n_perceptrons=2)
        ann_mean = _targets(model, value=1.0)

        def partial_means():
            full = model.cascade_means()
            del full[1]  # perceptron 1 has no recorded cascade value
            return full

        before1 = model.get_perceptrons()[1].layer.bias.detach().clone()
        dfq_correct_biases(model, ann_mean, partial_means, bias_iters=5, eta=0.7)
        torch.testing.assert_close(
            model.get_perceptrons()[1].layer.bias.detach(), before1, rtol=0, atol=0,
        )
