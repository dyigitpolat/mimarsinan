"""Per-channel trainable theta co-training in the TTFS-cycle tuner (default-OFF).

With ``ttfs_theta_cotrain=True`` (cascaded only) the tuner promotes each non-encoding
perceptron's ``activation_scale`` to a per-output-channel ``requires_grad`` Parameter
so the deployed-cascade fine-tune co-optimises the firing-gain (theta) WITH the
weights — the near-lossless cascaded recipe's key lever. Flag-OFF must be byte-
identical to today; synchronized ignores the flag; it is mutually exclusive with the
per-depth gain-correction ramp (both manage theta).
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
)


def _make(tmp_path, *, theta_cotrain, schedule="cascaded", gain_ramp=False, blend=False):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["simulation_steps"] = 16
    cfg["ttfs_theta_cotrain"] = theta_cotrain
    cfg["ttfs_gain_correction_ramp"] = gain_ramp
    cfg["ttfs_genuine_blend_ramp"] = blend
    if blend:
        cfg["ttfs_distmatch_bias_iters"] = 3
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner, model


class TestFlagOff:
    def test_default_off(self, tmp_path):
        tuner, model = _make(tmp_path, theta_cotrain=False)
        assert tuner._theta_cotrain is False
        for p in model.get_perceptrons():
            assert p.activation_scale.dim() == 0
            assert not p.activation_scale.requires_grad


class TestFlagOnCascaded:
    def test_non_encoding_theta_per_channel_trainable(self, tmp_path):
        tuner, model = _make(tmp_path, theta_cotrain=True)
        assert tuner._theta_cotrain is True
        promoted = 0
        for p in model.get_perceptrons():
            if getattr(p, "is_encoding_layer", False):
                assert p.activation_scale.dim() == 0
                continue
            assert p.activation_scale.requires_grad
            assert p.activation_scale.dim() == 1
            assert p.activation_scale.numel() == p.layer.weight.shape[0]
            promoted += 1
        assert promoted >= 1

    def test_theta_in_trainable_model_params(self, tmp_path):
        tuner, model = _make(tmp_path, theta_cotrain=True)
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        for p in model.get_perceptrons():
            if getattr(p, "is_encoding_layer", False):
                continue
            assert id(p.activation_scale) in trainable

    def test_stats_recorded(self, tmp_path):
        tuner, _ = _make(tmp_path, theta_cotrain=True)
        assert tuner._theta_cotrain_stats is not None
        assert tuner._theta_cotrain_stats["n_theta"] >= 1


class TestSynchronizedIgnores:
    def test_synchronized_inert(self, tmp_path):
        tuner, model = _make(tmp_path, theta_cotrain=True, schedule="synchronized")
        assert tuner._theta_cotrain is False
        for p in model.get_perceptrons():
            assert p.activation_scale.dim() == 0


class TestMutualExclusionWithGainRamp:
    def test_gain_ramp_wins_theta_cotrain_off(self, tmp_path):
        """Both manage activation_scale; the per-depth gain ramp takes precedence and
        theta-cotrain is disabled (no per-channel promotion that the gain ramp's
        scalar set_activation_scale would clobber)."""
        tuner, model = _make(tmp_path, theta_cotrain=True, gain_ramp=True)
        assert tuner._gain_ramp is True
        assert tuner._theta_cotrain is False
        for p in model.get_perceptrons():
            assert p.activation_scale.dim() == 0


class TestComposesWithGenuineBlend:
    def test_theta_cotrain_with_blend_ramp(self, tmp_path):
        tuner, model = _make(tmp_path, theta_cotrain=True, blend=True)
        assert tuner._theta_cotrain is True
        for p in model.get_perceptrons():
            if getattr(p, "is_encoding_layer", False):
                continue
            assert p.activation_scale.requires_grad and p.activation_scale.dim() == 1


def _run_step(cfg, tmp_path):
    from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
        TTFSCycleAdaptationStep,
    )

    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    am = AdaptationManager()
    pipeline.seed("model", model, step_name="Activation Quantization")
    pipeline.seed("adaptation_manager", am, step_name="Activation Quantization")
    step = TTFSCycleAdaptationStep(pipeline)
    step.name = "TTFS Cycle Fine-Tuning"
    pipeline.prepare_step(step)
    step.run()
    return step, model


class TestEndToEnd:
    def _cfg(self, tmp_path):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 16
        cfg["ttfs_theta_cotrain"] = True
        return cfg

    def test_step_runs_and_theta_is_trained(self, tmp_path):
        torch.manual_seed(5)
        cfg = self._cfg(tmp_path)
        step, model = _run_step(cfg, tmp_path)
        result = step.validate()
        assert isinstance(result, float) and 0.0 <= result <= 1.0
        # theta survived as a per-channel trainable param on the deployed model
        for p in model.get_perceptrons():
            if getattr(p, "is_encoding_layer", False):
                continue
            assert p.activation_scale.requires_grad and p.activation_scale.dim() == 1

    def test_theta_actually_updated_by_training(self, tmp_path):
        """The co-trained theta must MOVE during the step (the optimiser includes it
        and the gradient reaches it) — not stay at its promoted init."""
        torch.manual_seed(6)
        cfg = self._cfg(tmp_path)
        # capture promoted init by constructing a tuner the same way
        tuner, model0 = _make(tmp_path, theta_cotrain=True)
        init = [p.activation_scale.detach().clone() for p in model0.get_perceptrons()
                if not getattr(p, "is_encoding_layer", False)]
        step, model = _run_step(cfg, tmp_path)
        after = [p.activation_scale.detach() for p in model.get_perceptrons()
                 if not getattr(p, "is_encoding_layer", False)]
        moved = any(a.shape == b.shape and torch.any(a != b)
                    for a, b in zip(init, after))
        assert moved, "co-trained theta did not change during the step"
