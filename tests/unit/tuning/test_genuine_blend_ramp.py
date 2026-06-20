"""Genuine teacher->cascade blend ramp for cascaded ttfs_cycle (default-OFF).

With ``ttfs_genuine_blend_ramp=True`` (cascaded only) the deployed single-spike
cascade is calibrated to the teacher ANN's activation distribution
(:func:`match_activation_distributions`), then a ``BlendedGenuineForward`` is
installed as ``model.forward`` for the WHOLE ramp:

    out = (1 - rate) * teacher(x) + rate * genuine(x)

The committed rate drives ``forward.rate`` live (a ``GenuineBlendAxis``), so
``rate=0`` reads the frozen continuous teacher exactly and ``rate=1`` runs the
genuine single-spike cascade exactly (bit-identical to a freshly built cascade).
KD recovery against the same frozen teacher trains the genuine branch during the
ramp. At finalize the PURE genuine ``_SegmentSpikeForward`` is deployed (so the
finalize cliff -> 0 by construction; the teacher is dropped).

These are MECHANISM tests; the empirical accuracy gate (genuine cascade 0.41 ->
0.9355 on the real model) is a separate full run. Flag-OFF behavior must stay
byte-identical to the shipping value-domain proxy ramp (the genuine-blend flag is
mutually exclusive with the genuine-annealed flag; blend wins).
"""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.blended_genuine_forward import (
    BlendedGenuineForward,
)
from mimarsinan.tuning.axes.blend_axis import BlendAxis, GenuineBlendAxis
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.perceptron_rate import rebuild_activations, set_blend_rate
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
    _SegmentSpikeForward,
)


def _make_pipeline(tmp_path, *, schedule="cascaded", blend=True, annealed=False):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 16
    cfg["ttfs_genuine_blend_ramp"] = blend
    cfg["ttfs_genuine_annealed_ramp"] = annealed
    # Tiny calibration loops keep the unit test fast and deterministic.
    cfg["ttfs_distmatch_bias_iters"] = 3
    cfg["ttfs_distmatch_bias_eta"] = 0.7
    cfg["ttfs_distmatch_quantile"] = 0.99
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    return pipeline


def _make_tuner(tmp_path, *, schedule="cascaded", blend=True, annealed=False):
    pipeline = _make_pipeline(
        tmp_path, schedule=schedule, blend=blend, annealed=annealed
    )
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner, model, am


def _make_surrogate_tuner(tmp_path, *, surrogate, temp=1.0):
    pipeline = _make_pipeline(tmp_path)
    pipeline.config["ttfs_boundary_surrogate"] = surrogate
    pipeline.config["ttfs_boundary_surrogate_temp"] = temp
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner


class TestBoundarySurrogateFlag:
    """The offload-boundary STE flag threads the temperature into the genuine
    training forward (None = the historical severed contract)."""

    def test_default_off_is_none(self, tmp_path):
        tuner = _make_surrogate_tuner(tmp_path, surrogate=False)
        assert tuner._boundary_surrogate_temp is None

    def test_flag_sets_temp(self, tmp_path):
        tuner = _make_surrogate_tuner(tmp_path, surrogate=True, temp=2.0)
        assert tuner._boundary_surrogate_temp == 2.0

    def test_blended_forward_carries_temp(self, tmp_path):
        tuner = _make_surrogate_tuner(tmp_path, surrogate=True, temp=1.5)
        fwd = tuner._ramp_forward()
        assert fwd.boundary_surrogate_temp == 1.5

    def test_finalize_forward_carries_temp(self, tmp_path):
        tuner = _make_surrogate_tuner(tmp_path, surrogate=True, temp=1.5)
        fwd = tuner._finalize_forward_for(tuner.model)
        assert fwd.boundary_surrogate_temp == 1.5


class TestGainCorrectionFlag:
    """The per-depth gain correction fires during tuner construction (before the
    nodes are built) when ``ttfs_gain_correction`` is on, cascaded only."""

    def _make(self, tmp_path, *, on, rule="relative"):
        pipeline = _make_pipeline(tmp_path)
        pipeline.config["ttfs_gain_correction"] = on
        pipeline.config["ttfs_gain_correction_rule"] = rule
        model = make_tiny_supermodel()
        before = [float(p.activation_scale) for p in model.get_perceptrons()]
        am = AdaptationManager()
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5,
            lr=pipeline.config["lr"], adaptation_manager=am,
        )
        return tuner, before

    def test_default_off_no_correction(self, tmp_path):
        tuner, _ = self._make(tmp_path, on=False)
        assert tuner._gain_correction_stats is None

    def test_flag_applies_correction(self, tmp_path):
        tuner, before = self._make(tmp_path, on=True)
        assert tuner._gain_correction_stats is not None
        assert tuner._gain_correction_stats["n_corrected"] >= 1
        after = [float(p.activation_scale) for p in tuner.model.get_perceptrons()]
        # depth-0 unchanged, a deeper layer shrunk (relative rule)
        assert after[0] == pytest.approx(before[0])
        assert any(a < b - 1e-9 for a, b in zip(after, before))


class TestGainCorrectionRamp:
    """Rate-gated gain correction: scales ramp base -> base*g_d with _set_rate, so
    the calibration co-adapts with the KD blend (rate 0 = base, rate 1 = full)."""

    def _make(self, tmp_path, *, rule="relative"):
        # Plain cascaded value-domain blend (no genuine-blend distmatch, which would
        # ALSO set activation_scale and compete with the gain ramp): the gain ramp is
        # the sole scale operation, co-ramping with the blend rate.
        pipeline = _make_pipeline(tmp_path, blend=False)
        pipeline.config["ttfs_gain_correction_ramp"] = True
        pipeline.config["ttfs_gain_correction_rule"] = rule
        model = make_tiny_supermodel()
        base = [float(p.activation_scale) for p in model.get_perceptrons()]
        am = AdaptationManager()
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5,
            lr=pipeline.config["lr"], adaptation_manager=am,
        )
        return tuner, base

    def test_ramp_flag_recognized_and_starts_at_base(self, tmp_path):
        tuner, base = self._make(tmp_path)
        assert tuner._gain_ramp is True
        # after construction (_apply_gain_at_rate(0.0)) scales are at base
        now = [float(p.activation_scale) for p in tuner.model.get_perceptrons()]
        assert now == pytest.approx(base)

    def test_set_rate_ramps_scales(self, tmp_path):
        tuner, base = self._make(tmp_path)
        factors = tuner._gain_ramp_factors
        ps = list(tuner.model.get_perceptrons())
        tuner._set_rate(1.0)
        for p, b in zip(ps, base):
            assert float(p.activation_scale) == pytest.approx(b * factors[id(p)], rel=1e-5)
        tuner._set_rate(0.0)
        for p, b in zip(ps, base):
            assert float(p.activation_scale) == pytest.approx(b, rel=1e-5)


def _x(pipeline, n=3):
    return torch.randn(n, *pipeline.config["input_shape"])


# ── Flag ON: the ramp forward is a teacher<->genuine blend ────────────────────


class TestRampForwardIsBlend:
    def test_ramp_forward_is_blended_genuine(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        fwd = tuner._ramp_forward()
        assert isinstance(fwd, BlendedGenuineForward)
        assert fwd.model is model
        assert fwd.teacher is tuner._teacher
        assert fwd.T == tuner._T
        assert fwd.rate == pytest.approx(0.0)

    def test_blended_forward_installed_on_model(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        installed = model.__dict__.get("forward")
        assert isinstance(installed, BlendedGenuineForward), (
            "the genuine blend ramp must install BlendedGenuineForward as "
            "model.forward for the WHOLE ramp"
        )

    def test_rate_zero_is_the_frozen_teacher_exactly(self, tmp_path):
        torch.manual_seed(3)
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        tuner._set_rate(0.0)
        x = _x(tuner.pipeline)
        with torch.no_grad():
            got = model(x)
            expected = tuner._teacher(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_rate_one_is_a_freshly_built_genuine_cascade_exactly(self, tmp_path):
        torch.manual_seed(4)
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        tuner._set_rate(1.0)
        x = _x(tuner.pipeline)

        fresh = _SegmentSpikeForward(model, tuner._T)
        with torch.no_grad():
            got = model(x)
            expected = fresh(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


# ── Flag ON: genuine controller perf — the LR is found once, never re-found ───


class TestGenuineLrCachedOnce:
    def test_genuine_ramp_does_not_invalidate_lr_cache(self, tmp_path):
        """The LR finder is the genuine controller's dominant cost (~55s/call on
        the cascade); the LR is stable across the blend ramp, so the genuine path
        caches it once and never re-finds (``_invalidate_lr_cache`` is a no-op)."""
        tuner, _, _ = _make_tuner(tmp_path, blend=True)
        tuner._cached_lr = 0.0042
        tuner._invalidate_lr_cache()
        assert tuner._cached_lr == 0.0042, "genuine ramp must NOT drop the cached LR"

    def test_value_domain_ramp_still_invalidates(self, tmp_path):
        """Flag off (the value-domain proxy ramp): the base behavior is unchanged —
        the LR cache is still invalidated (golden-safe for every other tuner)."""
        tuner, _, _ = _make_tuner(tmp_path, blend=False)
        tuner._cached_lr = 0.0042
        tuner._invalidate_lr_cache()
        assert tuner._cached_lr is None


# ── Flag ON: the axis drives the installed forward.rate ───────────────────────


class TestAxisDrivesForwardRate:
    def test_axis_is_genuine_blend(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, blend=True)
        assert isinstance(tuner._axis, GenuineBlendAxis)
        assert tuner._axis.name == "genuine_blend"

    def test_set_rate_drives_installed_forward_rate(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        installed = model.__dict__.get("forward")
        for r in (0.0, 0.25, 0.5, 0.75, 1.0):
            tuner._set_rate(r)
            assert installed.rate == pytest.approx(r)

    def test_extra_state_round_trips_forward_rate(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        installed = model.__dict__.get("forward")
        tuner._set_rate(0.6)
        state = tuner._get_extra_state()
        assert float(state) == pytest.approx(0.6)
        tuner._set_rate(0.2)
        assert installed.rate == pytest.approx(0.2)
        tuner._set_extra_state(state)
        assert installed.rate == pytest.approx(0.6)


# ── Flag ON: distribution matching was applied ────────────────────────────────


class TestDistributionMatchingApplied:
    def test_perceptron_scales_or_biases_changed_from_raw_mapped(self, tmp_path):
        """match_activation_distributions calibrates scale-aware [0,1] boundaries
        and DFQ-corrects per-neuron biases; the deployed cascade must differ from
        the raw mapped model (otherwise calibration was a no-op)."""
        torch.manual_seed(8)
        pipeline = _make_pipeline(tmp_path, blend=True)
        model = make_tiny_supermodel()
        raw_scales = [
            float(p.activation_scale) for p in model.get_perceptrons()
        ]
        raw_biases = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        am = AdaptationManager()
        TTFSCycleAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5,
            lr=pipeline.config["lr"], adaptation_manager=am,
        )
        new_scales = [
            float(p.activation_scale) for p in model.get_perceptrons()
        ]
        new_biases = [
            p.layer.bias.detach().clone()
            for p in model.get_perceptrons()
            if getattr(p.layer, "bias", None) is not None
        ]
        scales_changed = any(
            abs(a - b) > 1e-6 for a, b in zip(raw_scales, new_scales)
        )
        biases_changed = any(
            not torch.equal(a, b) for a, b in zip(raw_biases, new_biases)
        )
        assert scales_changed or biases_changed, (
            "match_activation_distributions must mutate the deployed cascade "
            "(boundaries and/or DFQ biases) away from the raw mapped model"
        )

    def test_calibration_stats_recorded(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, blend=True)
        stats = tuner._distmatch_stats
        assert stats is not None
        assert stats["bias_iters"] == 3
        assert stats["quantile"] == pytest.approx(0.99)


# ── Flag ON: KD recovery trains the genuine branch ────────────────────────────


class TestKDRecoveryTrainsGenuine:
    def test_loss_is_differentiable_into_the_model(self, tmp_path):
        """The recovery loss (KD vs teacher + CE) on the blend forward must produce
        a gradient on the model's parameters via the genuine branch."""
        torch.manual_seed(11)
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        tuner._set_rate(0.5)
        x = _x(tuner.pipeline)
        y = torch.randint(0, tuner.pipeline.config["num_classes"], (x.shape[0],))
        param = next(p for p in model.parameters() if p.requires_grad)

        model.zero_grad(set_to_none=True)
        loss = tuner.trainer.loss_function(model, x, y)
        loss.backward()
        assert param.grad is not None
        assert torch.any(param.grad != 0)

    def test_loss_includes_pure_genuine_ce_term(self, tmp_path):
        """The validated recipe adds a small CE on the PURE genuine logits so the
        ramp pulls the r=1 endpoint up. At rate 0 the blend logits equal the
        teacher exactly, so the only y-coupled signal is the genuine-CE term —
        a nonzero gradient at rate 0 proves the genuine-CE term is present."""
        torch.manual_seed(12)
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        tuner._set_rate(0.0)  # blend logits == frozen teacher (no model gradient there)
        x = _x(tuner.pipeline)
        y = torch.randint(0, tuner.pipeline.config["num_classes"], (x.shape[0],))
        param = next(p for p in model.parameters() if p.requires_grad)

        model.zero_grad(set_to_none=True)
        loss = tuner.trainer.loss_function(model, x, y)
        loss.backward()
        assert param.grad is not None and torch.any(param.grad != 0), (
            "at rate 0 the blend logits are the frozen teacher; a model gradient "
            "can only come from the pure-genuine CE term"
        )


# ── Flag ON: the genuine-CE loss resolves via a tuner-owned reference ─────────


class TestLossDeFragilized:
    """The genuine-CE term resolves the installed ``BlendedGenuineForward`` through
    a tuner-owned reference + the public ``genuine_logits``, never by introspecting
    ``model.__dict__['forward']`` and a private ``_genuine`` attr (review Part F)."""

    def test_tuner_owns_installed_blend_forward(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        assert isinstance(tuner._blend_forward, BlendedGenuineForward)
        assert tuner._blend_forward is model.__dict__.get("forward")

    def test_loss_resolves_genuine_via_owned_reference(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, blend=True)
        loss = tuner.trainer.loss_function
        assert loss._blend_forward() is tuner._blend_forward

    def test_loss_robust_to_install_convention_change(self, tmp_path):
        """The old code read ``model.__dict__['forward']``; if the install
        convention changed, the genuine-CE term silently vanished. The owned
        reference is immune — swapping ``model.forward`` does not change what the
        loss resolves."""
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        owned = tuner._blend_forward
        model.forward = object()  # simulate a changed install convention
        assert tuner.trainer.loss_function._blend_forward() is owned

    def test_remove_forward_clears_owned_reference(self, tmp_path):
        """After the blend is removed (finalize/stabilization run the pure
        cascade), ``model(x)`` IS genuine, so the provider must return ``None`` and
        the loss must skip the extra genuine-CE term (no double-count)."""
        tuner, _, _ = _make_tuner(tmp_path, blend=True)
        assert tuner._blend_forward is not None
        tuner._remove_forward()
        assert tuner._blend_forward is None
        assert tuner.trainer.loss_function._blend_forward() is None


# ── Flag ON: finalize deploys the PURE genuine cascade ────────────────────────


class TestFinalizeIsPureGenuine:
    def test_finalize_forward_is_segment_spike(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=True)
        fwd = tuner._finalize_forward()
        assert isinstance(fwd, _SegmentSpikeForward)
        assert fwd.model is model

    def test_finalize_forward_has_no_teacher(self, tmp_path):
        """The finalized/deployed forward is the genuine cascade, not the blend —
        the teacher must be dropped at finalize (cliff -> 0 by construction)."""
        tuner, _, _ = _make_tuner(tmp_path, blend=True)
        fwd = tuner._finalize_forward()
        assert not isinstance(fwd, BlendedGenuineForward)
        assert not hasattr(fwd, "teacher")


# ── Flag ON: end-to-end run executes cleanly + leaves the genuine cascade ─────


def _run_step(pipeline):
    from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
        TTFSCycleAdaptationStep,
    )

    model = make_tiny_supermodel()
    am = AdaptationManager()
    pipeline.seed("model", model, step_name="Activation Quantization")
    pipeline.seed("adaptation_manager", am, step_name="Activation Quantization")
    step = TTFSCycleAdaptationStep(pipeline)
    step.name = "TTFS Cycle Fine-Tuning"
    pipeline.prepare_step(step)
    step.run()
    return step, model, am


class TestEndToEndRun:
    def test_step_runs_and_returns_float(self, tmp_path):
        torch.manual_seed(9)
        pipeline = _make_pipeline(tmp_path, blend=True)
        step, _, _ = _run_step(pipeline)
        result = step.validate()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_deployed_forward_persists_genuine_cascade(self, tmp_path):
        torch.manual_seed(10)
        pipeline = _make_pipeline(tmp_path, blend=True)
        step, model, _ = _run_step(pipeline)
        installed = model.__dict__.get("forward")
        assert isinstance(installed, _SegmentSpikeForward)
        assert not isinstance(installed, BlendedGenuineForward)

        T = int(pipeline.config["simulation_steps"])
        x = _x(pipeline)
        fresh = _SegmentSpikeForward(model, T)
        with torch.no_grad():
            torch.testing.assert_close(model(x), fresh(x), rtol=0, atol=0)


# ── Flag precedence: blend wins over annealed ─────────────────────────────────


class TestBlendWinsOverAnnealed:
    def test_blend_takes_precedence_over_annealed(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=True, annealed=True)
        assert tuner._genuine_blend_ramp is True
        assert tuner._genuine_annealed_ramp is False
        assert isinstance(tuner._axis, GenuineBlendAxis)
        assert isinstance(model.__dict__.get("forward"), BlendedGenuineForward)


# ── Flag OFF (default): value-domain proxy ramp byte-identical to today ────────


class TestFlagOffUnchanged:
    def test_genuine_blend_ramp_disabled_by_default(self, tmp_path):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 16
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.5
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5,
            lr=pipeline.config["lr"], adaptation_manager=am,
        )
        assert tuner._genuine_blend_ramp is False

    def test_ramp_forward_is_none(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, blend=False)
        assert tuner._ramp_forward() is None

    def test_finalize_forward_is_segment_spike(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=False)
        fwd = tuner._finalize_forward()
        assert isinstance(fwd, _SegmentSpikeForward)
        assert fwd.model is model

    def test_axis_is_plain_blend(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, blend=False)
        assert type(tuner._axis) is BlendAxis

    def test_base_activation_is_blend(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=False)
        for p in model.get_perceptrons():
            assert not isinstance(p.base_activation, TTFSActivation)
            assert hasattr(p.base_activation, "rate")
            assert isinstance(p.base_activation.target_activation, TTFSActivation)

    def test_no_forward_installed_during_ramp(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, blend=False)
        assert "forward" not in model.__dict__

    def test_no_distmatch_stats(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, blend=False)
        assert tuner._distmatch_stats is None


# ── Axis unit (no tuner): set_rate drives the installed BlendedGenuineForward ─


class _FakeModel(torch.nn.Module):
    """Minimal model carrying an installed BlendedGenuineForward instance."""

    def __init__(self):
        super().__init__()
        self.forward = _FakeBlended()

    def get_perceptrons(self):
        return []


class _FakeBlended:
    def __init__(self):
        self.rate = 0.0


class TestGenuineBlendAxisUnit:
    def test_set_rate_sets_installed_forward_rate(self):
        model = _FakeModel()
        axis = GenuineBlendAxis()
        axis.attach(model, None, {})
        axis.set_rate(0.42)
        assert model.__dict__["forward"].rate == pytest.approx(0.42)

    def test_extra_state_carries_forward_rate(self):
        model = _FakeModel()
        axis = GenuineBlendAxis()
        axis.attach(model, None, {})
        axis.set_rate(0.7)
        assert float(axis.get_extra_state()) == pytest.approx(0.7)
        axis.set_rate(0.1)
        axis.set_extra_state(0.7)
        assert model.__dict__["forward"].rate == pytest.approx(0.7)

    def test_set_rate_is_noop_when_no_forward_installed(self):
        class _Bare(torch.nn.Module):
            def get_perceptrons(self):
                return []

        axis = GenuineBlendAxis()
        axis.attach(_Bare(), None, {})
        axis.set_rate(1.0)  # must not raise


# ── Synchronized + flag ON: flag ignored ──────────────────────────────────────


class TestSynchronizedIgnoresFlag:
    def test_synchronized_ramp_forward_none_even_with_flag(self, tmp_path):
        tuner, model, _ = _make_tuner(tmp_path, schedule="synchronized", blend=True)
        assert tuner._genuine_blend_ramp is False
        assert tuner._ramp_forward() is None
        assert "forward" not in model.__dict__

    def test_synchronized_axis_is_plain_blend_even_with_flag(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, schedule="synchronized", blend=True)
        assert type(tuner._axis) is BlendAxis
