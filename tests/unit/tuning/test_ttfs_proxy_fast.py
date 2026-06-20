"""Fast PROXY ramp for cascaded ttfs_cycle (opt-in, default OFF).

With ``ttfs_blend_fast=True`` (cascaded, NOT a genuine ramp) the VALUE-DOMAIN
BlendActivation ramp runs through the ONE orchestrator's ``fixed_ladder`` policy +
a post-finalize bounded stabilization on the deployed genuine cascade
(``_SegmentSpikeForward``) — the LIF fast-fold pattern applied to TTFS's
better-accuracy (proxy) path. Uses the installed KD loss (base ``_fast_loss``, no
genuine-CE — there is no BlendedGenuineForward); the stabilization closes the
proxy↔genuine cliff. Default-OFF: the controller proxy ramp path is unchanged.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.axes.blend_axis import BlendAxis
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
    _SegmentSpikeForward,
)


def _make_tuner(tmp_path, *, fast=True, steps_per_rate=3, stabilize=0, rates=None):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "cascaded"
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 16
    cfg["ttfs_blend_fast"] = fast
    cfg["ttfs_blend_fast_steps_per_rate"] = steps_per_rate
    cfg["ttfs_blend_fast_stabilize_steps"] = stabilize
    if rates is not None:
        cfg["ttfs_blend_fast_rates"] = rates
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=am,
    )
    return tuner, model, am


class TestProxyFastFlag:
    def test_proxy_fast_recognized(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=True)
        assert tuner._proxy_fast is True
        assert tuner._fixed_ladder_policy is True
        # proxy is NOT a genuine ramp: keeps the value-domain BlendAxis + KD loss
        assert tuner._genuine_blend_ramp is False
        assert type(tuner._axis) is BlendAxis

    def test_default_off(self, tmp_path):
        tuner, _, _ = _make_tuner(tmp_path, fast=False)
        assert tuner._proxy_fast is False
        assert tuner._fixed_ladder_policy is False

    def test_proxy_fast_loss_is_kd_not_genuine_ce(self, tmp_path):
        """No BlendedGenuineForward on the proxy path → _fast_loss is the installed
        KD loss (base), not the genuine-CE objective."""
        tuner, _, _ = _make_tuner(tmp_path, fast=True)
        assert tuner._blend_forward is None
        assert tuner._installed_genuine_branch() is None


class TestProxyFastRun:
    def test_run_commits_one_and_deploys_pure_cascade(self, tmp_path):
        torch.manual_seed(0)
        tuner, model, _ = _make_tuner(
            tmp_path, fast=True, steps_per_rate=3, stabilize=5, rates=[0.5, 1.0],
        )
        tuner.run()
        assert tuner._committed_rate == pytest.approx(1.0)
        assert tuner._fast_blend_path is True
        assert len(tuner._cycle_log) == len(tuner._fixed_ladder_rates)
        installed = model.__dict__.get("forward")
        assert isinstance(installed, _SegmentSpikeForward)

    def test_run_returns_metric(self, tmp_path):
        torch.manual_seed(0)
        tuner, _, _ = _make_tuner(tmp_path, fast=True, steps_per_rate=2, stabilize=4)
        result = tuner.run()
        assert isinstance(result, float) and 0.0 <= result <= 1.0
