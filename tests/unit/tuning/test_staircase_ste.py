"""Staircase-backward STE in the TTFS-cycle tuner (default-OFF).

With ``ttfs_staircase_ste=True`` (cascaded only) the genuine-cascade training loss
uses a straight-through estimator: forward value == the genuine single-spike cascade
(the exact deploy path), backward == a hedge of the CLEAN complete-sum staircase
gradient + the genuine surrogate (``ttfs_ste_mix``, default 0.5). This is the fix for
the deep high-S surrogate-gradient plateau (research: lossless cascaded TTFS in <2min).
Flag-OFF must be byte-identical; synchronized ignores the flag.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
    _StaircaseSteKDLoss,
    _SegmentSpikeForward,
)


def _make(tmp_path, *, ste, mix=0.5, schedule="cascaded"):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["simulation_steps"] = 16
    cfg["ttfs_staircase_ste"] = ste
    cfg["ttfs_ste_mix"] = mix
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner, model


class TestFlag:
    def test_default_off(self, tmp_path):
        tuner, _ = _make(tmp_path, ste=False)
        assert tuner._staircase_ste is False
        assert type(tuner.trainer.loss_function) is not _StaircaseSteKDLoss

    def test_flag_on_cascaded(self, tmp_path):
        tuner, model = _make(tmp_path, ste=True, mix=0.5)
        assert tuner._staircase_ste is True
        assert tuner._ste_mix == 0.5
        # the genuine cascade forward is installed for the whole ramp
        assert isinstance(model.__dict__.get("forward"), _SegmentSpikeForward)
        assert isinstance(tuner.trainer.loss_function, _StaircaseSteKDLoss)

    def test_synchronized_ignores(self, tmp_path):
        tuner, _ = _make(tmp_path, ste=True, schedule="synchronized")
        assert tuner._staircase_ste is False


class TestSteLoss:
    def test_forward_value_is_genuine(self, tmp_path):
        """STE forward value must equal the pure genuine cascade (so the committed/
        deployed metric is the genuine path); only the BACKWARD differs."""
        torch.manual_seed(3)
        tuner, model = _make(tmp_path, ste=True, mix=0.5)
        x = torch.randn(3, *tuner.pipeline.config["input_shape"])
        loss = tuner.trainer.loss_function
        fwd = model.__dict__["forward"]
        with torch.no_grad():
            ste = loss._ste_logits(fwd, x)
            genuine = fwd(x)
        torch.testing.assert_close(ste, genuine, rtol=0, atol=0)

    def test_gradient_flows_into_model(self, tmp_path):
        torch.manual_seed(4)
        tuner, model = _make(tmp_path, ste=True, mix=0.5)
        x = torch.randn(3, *tuner.pipeline.config["input_shape"])
        y = torch.randint(0, tuner.pipeline.config["num_classes"], (3,))
        param = next(p for p in model.parameters() if p.requires_grad)
        model.zero_grad(set_to_none=True)
        tuner.trainer.loss_function(model, x, y).backward()
        assert param.grad is not None and torch.any(param.grad != 0)

    def test_mix_changes_backward_not_forward(self, tmp_path):
        """Different mix -> same forward value (genuine), different gradient."""
        torch.manual_seed(5)

        def grad_for(mix):
            tuner, model = _make(tmp_path, ste=True, mix=mix)
            x = torch.randn(4, *tuner.pipeline.config["input_shape"])
            y = torch.randint(0, tuner.pipeline.config["num_classes"], (4,))
            p = next(p for p in model.parameters() if p.requires_grad)
            model.zero_grad(set_to_none=True)
            tuner.trainer.loss_function(model, x, y).backward()
            return p.grad.detach().clone()

        g0 = grad_for(0.0)
        g1 = grad_for(1.0)
        assert not torch.allclose(g0, g1), "mix must change the backward gradient"

    def test_cycle_accurate_restored_after_staircase(self, tmp_path):
        """Computing the staircase toggles cycle_accurate=False; it must be restored
        so the genuine forward stays correct."""
        from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

        tuner, model = _make(tmp_path, ste=True, mix=0.5)
        x = torch.randn(2, *tuner.pipeline.config["input_shape"])
        fwd = model.__dict__["forward"]
        nodes = [m for m in model.modules() if isinstance(m, TTFSActivation)]
        before = [n._cycle_accurate_mode for n in nodes]
        with torch.no_grad():
            tuner.trainer.loss_function._ste_logits(fwd, x)
        after = [n._cycle_accurate_mode for n in nodes]
        assert before == after


class TestSteRefine:
    """``ttfs_blend_fast_ste_refine`` (requires ``ttfs_blend_fast``): the proxy ramp
    revives the cascade (alive value domain → ramp in), then the post-finalize
    stabilize refines the revived DEPLOYED cascade with the STE loss instead of plain
    KD — the two-stage revive→refine recipe (direct STE on the dead cold cascade is
    chance; the proxy revival makes the STE a refinement lever on an alive cascade)."""

    def _make(self, tmp_path, *, ste_refine=True, proxy=True, stab=50):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 16
        cfg["ttfs_blend_fast"] = proxy
        cfg["ttfs_blend_fast_ste_refine"] = ste_refine
        cfg["ttfs_blend_fast_stabilize_steps"] = stab
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.5
        model = make_tiny_supermodel()
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5,
            lr=pipeline.config["lr"], adaptation_manager=AdaptationManager(),
        )
        return tuner, model

    def test_flag_requires_proxy_fast(self, tmp_path):
        tuner, _ = self._make(tmp_path, ste_refine=True, proxy=False)
        assert tuner._staircase_ste_refine is False, "ste_refine is inert without proxy_fast"
        tuner, _ = self._make(tmp_path, ste_refine=True, proxy=True)
        assert tuner._staircase_ste_refine is True

    def test_hook_passes_ste_loss_when_refine(self, tmp_path):
        tuner, _ = self._make(tmp_path, ste_refine=True, stab=50)
        captured = {}
        tuner._fast_stabilize = lambda steps, loss_fn=None: captured.update(
            steps=steps, loss_fn=loss_fn,
        )
        tuner._post_stabilization_hook()
        assert captured["steps"] == 50
        assert captured["loss_fn"] is not None, "ste_refine must pass the STE loss_fn"

    def test_hook_uses_default_kd_when_off(self, tmp_path):
        tuner, _ = self._make(tmp_path, ste_refine=False, stab=50)
        captured = {}
        tuner._fast_stabilize = lambda steps, loss_fn=None: captured.update(
            steps=steps, loss_fn=loss_fn,
        )
        tuner._post_stabilization_hook()
        assert captured["loss_fn"] is None, "without ste_refine the stabilize stays plain KD"


class TestSteFast:
    """``ttfs_staircase_ste_fast`` (requires the STE): route the STE through a
    dedicated clean fixed-step loop (split-LR + progressive shallow->deep depth +
    cosine over ``ttfs_ste_steps``) instead of the rate-search controller, which caps
    the STE ~0.83 on MNIST. One fixed-ladder rung at rate 1.0; forward stays genuine."""

    def _make_fast(self, tmp_path, *, ste_fast=True, theta=False, steps=6,
                   init_frac=1.0 / 3.0, w_lr=2e-3, theta_lr=5e-2):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 16
        cfg["ttfs_staircase_ste"] = True
        cfg["ttfs_staircase_ste_fast"] = ste_fast
        cfg["ttfs_theta_cotrain"] = theta
        cfg["ttfs_ste_steps"] = steps
        cfg["ttfs_ste_w_lr"] = w_lr
        cfg["ttfs_ste_theta_lr"] = theta_lr
        cfg["ttfs_ste_init_frac"] = init_frac
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.5
        model = make_tiny_supermodel()
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5,
            lr=pipeline.config["lr"], adaptation_manager=AdaptationManager(),
        )
        return tuner, model

    def test_flag_enables_single_rung_ladder(self, tmp_path):
        tuner, _ = self._make_fast(tmp_path, steps=7)
        assert tuner._staircase_ste_fast is True
        assert tuner._fixed_ladder_policy is True
        assert tuner._fixed_ladder_rates == [1.0]
        assert tuner._fast_steps_per_rate == 7

    def test_off_without_flag_uses_controller(self, tmp_path):
        # STE without the fast flag rides the annealed-ramp controller (no ladder).
        tuner, _ = self._make_fast(tmp_path, ste_fast=False)
        assert tuner._staircase_ste_fast is False
        assert tuner._fixed_ladder_policy is False

    def test_weight_params_through_progressive(self, tmp_path):
        tuner, model = self._make_fast(tmp_path)
        perceptrons = model.get_perceptrons()
        n = len(perceptrons)
        tuner._weight_params_through(1)
        reqs = [p.layer.weight.requires_grad for p in perceptrons]
        assert reqs[0] is True and not any(reqs[1:]), "depth=1 trains only the shallowest"
        tuner._weight_params_through(n)
        assert all(p.layer.weight.requires_grad for p in perceptrons), "depth=n trains all"

    def test_run_commits_genuine_cascade_and_returns_metric(self, tmp_path):
        torch.manual_seed(0)
        tuner, model = self._make_fast(tmp_path, steps=6)
        result = tuner.run()
        assert isinstance(result, float) and 0.0 <= result <= 1.0
        assert tuner._committed_rate == pytest.approx(1.0)
        assert isinstance(model.__dict__.get("forward"), _SegmentSpikeForward)
        # the final schedule rung unfreezes every weight (deploy fine-tunes all layers)
        assert all(p.layer.weight.requires_grad for p in model.get_perceptrons())
        assert "fast_blend" in tuner._phase_seconds

    def test_run_records_one_commit(self, tmp_path):
        torch.manual_seed(0)
        tuner, _ = self._make_fast(tmp_path, steps=4)
        tuner.run()
        assert len(tuner._cycle_log) == 1
        assert tuner._cycle_log[0]["outcome"] == "commit"

    def test_split_lr_optimizer_groups(self, tmp_path, monkeypatch):
        captured = []
        real_adam = torch.optim.Adam

        def spy(groups, *a, **k):
            captured.append([(len(g["params"]), float(g["lr"])) for g in groups])
            return real_adam(groups, *a, **k)

        monkeypatch.setattr(torch.optim, "Adam", spy)
        torch.manual_seed(0)
        tuner, _ = self._make_fast(
            tmp_path, theta=True, steps=4, w_lr=2e-3, theta_lr=5e-2,
        )
        assert tuner._theta_cotrain_params, "theta co-train must populate the theta group"
        tuner.run()
        lrs = {lr for groups in captured for (_, lr) in groups}
        assert 2e-3 in lrs and 5e-2 in lrs, (
            f"split-LR: weights@2e-3 + theta@5e-2 expected, saw {sorted(lrs)}"
        )
