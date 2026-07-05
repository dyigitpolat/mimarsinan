"""Fast fixed-ladder LIF ramp (opt-in, default OFF).

With ``lif_blend_fast=True`` the LIF value-domain blend ramp runs through the ONE
orchestrator's ``fixed_ladder`` RateScheduler policy (one shared optimizer + spanning
warmup/cosine LR, KD recovery per rung, no per-cycle rollback/recovery/LR-find/
stabilization) instead of the greedy/bisect controller — the FAST analog of the slow
LIF controller ramp. It reuses ``KDBlendAdaptationTuner``'s shared fast machinery; the
per-step loss is the installed KD loss (base ``_fast_loss``) and there is no genuine
probe (LIF has no ``BlendedGenuineForward``). Flag-OFF is the unchanged controller path.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner


def _make_tuner(tmp_path, *, fast=True, steps_per_rate=3, rates=None, freeze_bn=False):
    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 4
    cfg["lif_blend_fast"] = fast
    cfg["fast_ladder_freeze_bn"] = freeze_bn
    cfg["lif_blend_fast_steps_per_rate"] = steps_per_rate
    if rates is not None:
        cfg["lif_blend_fast_rates"] = rates
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=am,
    )
    return tuner, model, am


class TestLifFastFlag:
    def test_flag_recognized(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True)
        assert tuner._fixed_ladder_policy is True

    def test_ladder_normalized_to_trailing_one(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True, rates=[0.5, 0.9])
        assert tuner._fixed_ladder_rates[-1] == pytest.approx(1.0)

    def test_default_off(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=False)
        assert tuner._fixed_ladder_policy is False


class TestLifFastRun:
    def test_run_commits_one_and_records_one_trace_per_rate(self, tmp_path, accepting_gate):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=3, rates=[0.5, 1.0])
        tuner.run()
        assert tuner._committed_rate == pytest.approx(1.0)
        assert tuner._fast_blend_path is True
        assert len(tuner._cycle_log) == len(tuner._fixed_ladder_rates)
        assert [e["outcome"] for e in tuner._cycle_log] == \
            ["commit"] * len(tuner._fixed_ladder_rates)
        assert tuner._fast_optimizer_steps == len(tuner._fixed_ladder_rates) * 3

    def test_run_returns_metric(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2)
        result = tuner.run()
        assert isinstance(result, float) and 0.0 <= result <= 1.0

    def test_fast_disables_stabilization(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True)
        assert tuner._stabilization_budget() == 0

    def test_bn_freeze_keeps_batchnorm_eval_during_fast_steps(self, tmp_path):
        tuner, model, _ = _make_tuner(
            tmp_path, fast=True, freeze_bn=True, steps_per_rate=1, rates=[1.0],
        )
        seen = []
        orig = tuner._fast_loss

        def wrapped(x, y):
            seen.extend(
                m.training
                for m in model.modules()
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            )
            return orig(x, y)

        tuner._fast_loss = wrapped
        tuner.run()
        assert seen
        assert not any(seen)

    def test_bn_freeze_default_off_keeps_historical_train_mode(self, tmp_path):
        tuner, model, _ = _make_tuner(
            tmp_path, fast=True, freeze_bn=False, steps_per_rate=1, rates=[1.0],
        )
        seen = []
        orig = tuner._fast_loss

        def wrapped(x, y):
            seen.extend(
                m.training
                for m in model.modules()
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            )
            return orig(x, y)

        tuner._fast_loss = wrapped
        tuner.run()
        assert seen
        assert any(seen)

    def test_rerun_resets_fast_scratch(self, tmp_path, accepting_gate):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2, rates=[0.5, 1.0])
        tuner.run()
        first = tuner._fast_optimizer
        tuner.run()
        assert tuner._fast_optimizer_steps == len(tuner._fixed_ladder_rates) * 2
        assert tuner._fast_optimizer is not first


class TestLifFastOffUnchanged:
    def test_controller_path_when_off(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=False)
        tuner.run()
        assert getattr(tuner, "_fast_blend_path", False) is False
        assert len(tuner._cycle_log) > 0


class TestLifDistmatchHook:
    """The DFQ teacher-distribution match fires from the post-stabilization hook
    when ``lif_distmatch`` is on (and only then), on the deployed cycle-accurate
    cascade."""

    def _make(self, tmp_path, *, distmatch):
        cfg = default_config()
        cfg["spiking_mode"] = "lif"
        cfg["firing_mode"] = "Default"
        cfg["thresholding_mode"] = "<"
        cfg["simulation_steps"] = 4
        cfg["cycle_accurate_lif_forward"] = True
        cfg["lif_blend_fast"] = True
        cfg["lif_blend_fast_steps_per_rate"] = 2
        cfg["lif_blend_fast_rates"] = [0.5, 1.0]
        cfg["lif_distmatch"] = distmatch
        cfg["lif_distmatch_bias_iters"] = 4
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.5
        model = make_tiny_supermodel()
        am = AdaptationManager()
        tuner = LIFAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
            adaptation_manager=am,
        )
        return tuner

    def test_distmatch_stats_populated_when_on(self, tmp_path):
        torch.manual_seed(0)
        tuner = self._make(tmp_path, distmatch=True)
        tuner.run()
        assert tuner._lif_distmatch_stats is not None
        assert "mean_gap_before" in tuner._lif_distmatch_stats
        assert "mean_gap_after" in tuner._lif_distmatch_stats

    def test_distmatch_not_run_when_off(self, tmp_path):
        torch.manual_seed(0)
        tuner = self._make(tmp_path, distmatch=False)
        tuner.run()
        assert tuner._lif_distmatch_stats is None
