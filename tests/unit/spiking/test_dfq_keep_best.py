"""W-CAL-3: the DFQ iteration loop is ratcheted over a deployed-behavior probe.

With a ``probe`` supplied, ``dfq_correct_biases`` measures the deployed probe
after every iteration, keeps the best-probe bias state, stops early on
sustained regression, and can never end worse than iteration-0. Without a
probe the loop is the classic (bit-identical) fixed-iteration form.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.spiking.dfq_bias_correction import dfq_correct_biases


class _FakePerceptron:
    def __init__(self, out_features):
        self.layer = nn.Linear(out_features, out_features)
        self.layer.bias.data.zero_()
        self.output_channel_axis = -1


class _FakeModel:
    """Decoded cascade mean for perceptron k is exactly its bias + offset."""

    def __init__(self, out_features=4, n_perceptrons=2, offset=0.5):
        self._perceptrons = [
            _FakePerceptron(out_features) for _ in range(n_perceptrons)
        ]
        self._offset = offset

    def get_perceptrons(self):
        return self._perceptrons

    def cascade_means(self):
        return {
            k: p.layer.bias.detach() + self._offset
            for k, p in enumerate(self._perceptrons)
        }

    def biases(self):
        return [p.layer.bias.detach().clone() for p in self._perceptrons]


def _targets(model, value=1.0):
    return {
        k: torch.full_like(p.layer.bias.detach(), value)
        for k, p in enumerate(model.get_perceptrons())
    }


def _scripted_probe(values):
    """A probe returning the scripted sequence: entry first, then one per iteration."""
    seq = iter(values)
    return lambda: next(seq)


def _run_plain_iters(n_iters, **model_kwargs):
    """Reference biases after ``n_iters`` classic (probe-free) DFQ iterations."""
    model = _FakeModel(**model_kwargs)
    dfq_correct_biases(
        model, _targets(model), model.cascade_means, bias_iters=n_iters, eta=0.7,
    )
    return model.biases()


class TestKeepBestRestoresBestIterate:
    def test_diverging_probe_restores_best_and_stops_early(self):
        model = _FakeModel()
        # entry 0.4; best at iter 1 (0.8); then sustained regression
        probe = _scripted_probe([0.4, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45])
        stats = dfq_correct_biases(
            model, _targets(model), model.cascade_means,
            bias_iters=15, eta=0.7, probe=probe, probe_patience=3,
        )
        assert stats["probe_best_iter"] == 1
        assert stats["probe_best"] == pytest.approx(0.8)
        assert stats["probe_entry"] == pytest.approx(0.4)
        # early stop: best at 1, then patience=3 regressing iterations
        assert stats["probe_iters_run"] == 4
        for got, want in zip(model.biases(), _run_plain_iters(1)):
            assert torch.equal(got, want)

    def test_entry_best_means_calibration_is_a_no_op(self):
        """Never worse than iteration-0: a correction that only regresses the
        deployed probe leaves the biases exactly at their entry state."""
        model = _FakeModel()
        entry = model.biases()
        probe = _scripted_probe([0.9, 0.5, 0.4, 0.3, 0.2])
        stats = dfq_correct_biases(
            model, _targets(model), model.cascade_means,
            bias_iters=15, eta=0.7, probe=probe, probe_patience=3,
        )
        assert stats["probe_best_iter"] == 0
        for got, want in zip(model.biases(), entry):
            assert torch.equal(got, want)

    def test_improving_probe_runs_all_iterations_and_keeps_last(self):
        model = _FakeModel()
        probe = _scripted_probe([0.1 + 0.05 * i for i in range(6)])
        stats = dfq_correct_biases(
            model, _targets(model), model.cascade_means,
            bias_iters=5, eta=0.7, probe=probe, probe_patience=3,
        )
        assert stats["probe_iters_run"] == 5
        assert stats["probe_best_iter"] == 5
        for got, want in zip(model.biases(), _run_plain_iters(5)):
            assert torch.equal(got, want)

    def test_no_patience_runs_all_iterations_but_still_keeps_best(self):
        model = _FakeModel()
        probe = _scripted_probe([0.4, 0.8] + [0.3] * 14)
        stats = dfq_correct_biases(
            model, _targets(model), model.cascade_means,
            bias_iters=15, eta=0.7, probe=probe, probe_patience=None,
        )
        assert stats["probe_iters_run"] == 15
        assert stats["probe_best_iter"] == 1
        for got, want in zip(model.biases(), _run_plain_iters(1)):
            assert torch.equal(got, want)

    def test_probe_curve_is_reported(self):
        model = _FakeModel()
        probe = _scripted_probe([0.4, 0.6, 0.7])
        stats = dfq_correct_biases(
            model, _targets(model), model.cascade_means,
            bias_iters=2, eta=0.7, probe=probe, probe_patience=None,
        )
        assert stats["probe_curve"] == [
            pytest.approx(0.6), pytest.approx(0.7),
        ]


class TestNoProbeIsClassicLoop:
    def test_without_probe_biases_match_full_iteration_count(self):
        model = _FakeModel()
        stats = dfq_correct_biases(
            model, _targets(model), model.cascade_means, bias_iters=7, eta=0.7,
        )
        for got, want in zip(model.biases(), _run_plain_iters(7)):
            assert torch.equal(got, want)
        assert "probe_curve" not in stats


class TestDistmatchWiresDeployedProbe:
    """``run_teacher_distmatch`` supplies the deployed full-transform probe
    (the X2 gate's measurement) and the tuning-policy patience by default."""

    def test_probe_and_patience_threaded_to_matcher(self):
        from types import SimpleNamespace

        from mimarsinan.tuning.orchestration.blend_ramp import (
            run_teacher_distmatch,
        )
        from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY

        class _Reporter:
            def report(self, key, value):
                pass

        tuner = SimpleNamespace(
            model=object(), _teacher=object(), _T=16, name="Stub",
            pipeline=SimpleNamespace(reporter=_Reporter()),
            _calibration_inputs=lambda n_batches=8: "CAL_X",
        )
        seen = {}

        def matcher(model, teacher, cal_x, T, **kw):
            seen.update(kw)
            return {}

        run_teacher_distmatch(tuner, matcher, n_batches=8, bias_iters=3, eta=0.5)
        assert callable(seen["probe"])
        assert seen["probe_patience"] == TUNING_POLICY.dfq_keepbest_patience

    def test_probe_measures_the_tuners_deployed_full_transform(self, monkeypatch):
        from types import SimpleNamespace

        import mimarsinan.tuning.orchestration.blend_ramp as blend_ramp

        class _Reporter:
            def report(self, key, value):
                pass

        tuner = SimpleNamespace(
            model=object(), _teacher=object(), _T=16, name="Stub",
            pipeline=SimpleNamespace(reporter=_Reporter()),
            _calibration_inputs=lambda n_batches=8: "CAL_X",
        )
        monkeypatch.setattr(
            blend_ramp, "full_transform_measurement",
            lambda t: 0.75 if t is tuner else 0.0,
        )
        captured = {}

        def matcher(model, teacher, cal_x, T, **kw):
            captured["probe"] = kw["probe"]
            return {}

        blend_ramp.run_teacher_distmatch(tuner, matcher, n_batches=8)
        assert captured["probe"]() == pytest.approx(0.75)
