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
