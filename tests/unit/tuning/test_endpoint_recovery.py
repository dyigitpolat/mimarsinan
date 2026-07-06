"""The P1'' endpoint-recovery stage (MBH X3): bounded deployed-forward train-to-target.

One mechanism, N call sites: after a conversion tuner reaches rate 1.0 and
finalizes, ``run_endpoint_recovery`` trains the DEPLOYED-COMPOSITION forward
toward the pipeline D-hat high-water mark (fail-loud when absent), with budget =
the recipe's freed stabilize/ladder steps plus whatever this tuner's own gated
ladder left untrained, keep-best + early-stop inside the trainer loop, and an
fp32 entry guard so the stage never ends below entry. It replaces the old
``_fast_stabilize`` / open-ended stabilization at the LIF, TTFS-cycle, and sync
AQ endpoints.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.tuning.orchestration import dhat_highwater, endpoint_recovery
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.endpoint_recovery import (
    EndpointRecoveryReport,
    freed_ladder_steps,
    run_endpoint_recovery,
)
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine


def _lif_tuner(tmp_path, *, endpoint_steps=0, steps_per_rate=2, rates=(0.5, 1.0)):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = 4
    cfg["lif_blend_fast"] = True
    cfg["lif_blend_fast_steps_per_rate"] = steps_per_rate
    cfg["lif_blend_fast_rates"] = list(rates)
    cfg["endpoint_recovery_steps"] = endpoint_steps
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    return LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )


def _prepare_endpoint_scaffold(tuner):
    """The run-scope scratch the endpoint stage reads when driven directly."""
    tuner._phase_seconds = {}
    tuner._mbh_rung_index = -1
    tuner._mbh_gate_state = None
    tuner._rollback_tolerance = 0.0
    tuner._fast_optimizer_steps = 0


def _endpoint_lines(text):
    return [l for l in text.splitlines() if l.startswith("[MBH-ENDPOINT] ")]


class TestFreedLadderSteps:
    def test_untrained_ladder_frees_everything(self, tmp_path):
        tuner = _lif_tuner(tmp_path, steps_per_rate=3, rates=(0.5, 1.0))
        try:
            _prepare_endpoint_scaffold(tuner)
            assert freed_ladder_steps(tuner) == 2 * 3
        finally:
            tuner.close()

    def test_fully_trained_ladder_frees_nothing(self, tmp_path):
        tuner = _lif_tuner(tmp_path, steps_per_rate=3, rates=(0.5, 1.0))
        try:
            _prepare_endpoint_scaffold(tuner)
            tuner._fast_optimizer_steps = 6
            assert freed_ladder_steps(tuner) == 0
        finally:
            tuner.close()

    def test_overtrained_ladder_never_goes_negative(self, tmp_path):
        # Gate refinements can train MORE than the planned ladder.
        tuner = _lif_tuner(tmp_path, steps_per_rate=3, rates=(0.5, 1.0))
        try:
            _prepare_endpoint_scaffold(tuner)
            tuner._fast_optimizer_steps = 11
            assert freed_ladder_steps(tuner) == 0
        finally:
            tuner.close()

    def test_controller_tuner_frees_nothing(self):
        class _Bare:
            pass

        assert freed_ladder_steps(_Bare()) == 0


class TestFailLoudWithoutHighWater:
    def test_missing_mark_raises(self, tmp_path):
        tuner = _lif_tuner(tmp_path, endpoint_steps=5)
        try:
            _prepare_endpoint_scaffold(tuner)
            with pytest.raises(RuntimeError, match="high-water mark is absent"):
                run_endpoint_recovery(tuner, base_steps=5)
        finally:
            tuner.close()


class TestEngagement:
    def test_target_already_met_skips_training(self, tmp_path, monkeypatch, capsys):
        tuner = _lif_tuner(tmp_path)
        try:
            _prepare_endpoint_scaffold(tuner)
            dhat_highwater.observe(tuner.pipeline, 0.10)
            monkeypatch.setattr(
                endpoint_recovery, "_fp32_deployed_read", lambda t: 0.9,
            )
            calls = []
            monkeypatch.setattr(
                RecoveryEngine, "train_to_target",
                staticmethod(lambda *a, **k: calls.append(1)),
            )
            report = run_endpoint_recovery(tuner, base_steps=100)
            assert calls == []
            assert report.engaged is False
            assert report.reached is True
            assert report.steps_used == 0
            assert report.exit == pytest.approx(0.9)
        finally:
            tuner.close()
        assert len(_endpoint_lines(capsys.readouterr().out)) == 1

    def test_zero_budget_never_engages(self, tmp_path, monkeypatch):
        tuner = _lif_tuner(tmp_path)
        try:
            _prepare_endpoint_scaffold(tuner)
            tuner._fast_optimizer_steps = len(tuner._fixed_ladder_rates) * 2
            dhat_highwater.observe(tuner.pipeline, 0.99)
            monkeypatch.setattr(
                endpoint_recovery, "_fp32_deployed_read", lambda t: 0.2,
            )
            report = run_endpoint_recovery(tuner, base_steps=0)
            assert report.engaged is False and report.reached is False
        finally:
            tuner.close()

    def test_budget_is_base_plus_freed_and_target_is_the_mark(
        self, tmp_path, monkeypatch,
    ):
        tuner = _lif_tuner(tmp_path, steps_per_rate=3, rates=(0.5, 1.0))
        try:
            _prepare_endpoint_scaffold(tuner)
            tuner._fast_optimizer_steps = 2  # ladder left 4 of its 6 steps free
            dhat_highwater.observe(tuner.pipeline, 0.77)
            reads = iter([0.3, 0.5])
            monkeypatch.setattr(
                endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
            )
            seen = {}

            def fake_train(trainer, lr, target, *, max_steps, **kwargs):
                seen["target"] = target
                seen["max_steps"] = max_steps
                return 0.5, 7

            monkeypatch.setattr(
                RecoveryEngine, "train_to_target", staticmethod(fake_train),
            )
            report = run_endpoint_recovery(tuner, base_steps=300)
            assert seen["target"] == pytest.approx(0.77)
            assert seen["max_steps"] == 304
            assert report.budget_steps == 304
            assert report.steps_used == 7
            assert report.engaged is True
            assert report.reached is False
            assert report.exit == pytest.approx(0.5)
        finally:
            tuner.close()

    def test_exit_read_ratchets_the_mark(self, tmp_path, monkeypatch):
        tuner = _lif_tuner(tmp_path)
        try:
            _prepare_endpoint_scaffold(tuner)
            dhat_highwater.observe(tuner.pipeline, 0.4)
            reads = iter([0.3, 0.65])
            monkeypatch.setattr(
                endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
            )
            monkeypatch.setattr(
                RecoveryEngine, "train_to_target",
                staticmethod(lambda *a, **k: (0.65, 3)),
            )
            report = run_endpoint_recovery(tuner, base_steps=10)
            assert report.reached is True
            assert dhat_highwater.require(tuner.pipeline) == pytest.approx(0.65)
        finally:
            tuner.close()


class TestEndpointTargetFloor:
    """[5u] the endpoint target floor for bit-parity-lossless recipes.

    For a lossless mode every controller target anchors at a past deployed read,
    so preservation ≡ stagnation at the float envelope; the floor lets the
    endpoint chase the acceptance target. When the floor LIFTS the target above
    the high-water, the stage must also switch to the probe-validated geometry
    (lr ≤ 2e-3, cosine over the FULL funded budget) — the default patience
    window sits entirely inside the lr transient and would be sterile.
    """

    def _run(self, tmp_path, monkeypatch, *, floor, highwater, reads, base_steps=100):
        tuner = _lif_tuner(tmp_path)
        tuner.pipeline.config["endpoint_target_floor"] = floor
        _prepare_endpoint_scaffold(tuner)
        tuner._fast_optimizer_steps = len(tuner._fixed_ladder_rates) * 2
        dhat_highwater.observe(tuner.pipeline, highwater)
        read_iter = iter(reads)
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(read_iter),
        )
        seen = {}

        def fake_train(trainer, lr, target, *, max_steps, **kwargs):
            seen.update(kwargs, lr=lr, target=target, max_steps=max_steps)
            return reads[-1], 9

        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(fake_train),
        )
        try:
            report = run_endpoint_recovery(tuner, base_steps=base_steps)
        finally:
            tuner.close()
        return report, seen

    def test_floor_lifts_the_target_above_the_high_water(self, tmp_path, monkeypatch):
        report, seen = self._run(
            tmp_path, monkeypatch, floor=0.9, highwater=0.5, reads=[0.3, 0.6],
        )
        assert seen["target"] == pytest.approx(0.9)
        assert report.target == pytest.approx(0.9)
        assert report.floor_lifted is True

    def test_floor_below_high_water_keeps_the_mark_and_geometry(
        self, tmp_path, monkeypatch,
    ):
        # Sterile-window regression guard for the DEFAULT path: when the floor
        # does not lift the target, geometry is bit-identical to the pre-floor
        # stage (pipeline lr, min_steps=0, patience armed).
        report, seen = self._run(
            tmp_path, monkeypatch, floor=0.4, highwater=0.77, reads=[0.3, 0.6],
        )
        assert seen["target"] == pytest.approx(0.77)
        assert seen["min_steps"] == 0
        assert seen["lr"] == pytest.approx(0.001)
        assert report.floor_lifted is False

    def test_absent_floor_is_bit_identical_to_the_pre_floor_stage(
        self, tmp_path, monkeypatch,
    ):
        tuner = _lif_tuner(tmp_path)
        assert "endpoint_target_floor" not in tuner.pipeline.config
        _prepare_endpoint_scaffold(tuner)
        dhat_highwater.observe(tuner.pipeline, 0.77)
        reads = iter([0.3, 0.6])
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
        )
        seen = {}

        def fake_train(trainer, lr, target, *, max_steps, **kwargs):
            seen.update(kwargs, lr=lr, target=target)
            return 0.6, 9

        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(fake_train),
        )
        try:
            report = run_endpoint_recovery(tuner, base_steps=10)
        finally:
            tuner.close()
        assert seen["target"] == pytest.approx(0.77)
        assert seen["min_steps"] == 0
        assert report.floor_lifted is False and report.target_floor == 0.0

    def test_lifted_floor_disarms_the_sterile_patience_window(
        self, tmp_path, monkeypatch,
    ):
        # [5u geometry] min_steps == the full funded budget: stale checks accrued
        # inside the lr-transient dip must not stop the stage (keep-best still
        # floors at entry; the target-reach early exit remains).
        report, seen = self._run(
            tmp_path, monkeypatch, floor=0.9, highwater=0.5, reads=[0.3, 0.6],
            base_steps=300,
        )
        assert seen["max_steps"] == 300
        assert seen["min_steps"] == 300
        assert seen["cosine_decay"] is True

    def test_lifted_floor_funds_wall_headroom_not_steps(
        self, tmp_path, monkeypatch,
    ):
        # [5u amendment, fba wave] the contended transform-trainer rate is
        # ~30 steps/s (12k steps cost 402 s), so the funded budget is WALL:
        # the stage passes the policy's headroom cap; the default path never
        # carries a cap.
        report, seen = self._run(
            tmp_path, monkeypatch, floor=0.9, highwater=0.5, reads=[0.3, 0.6],
        )
        assert seen["max_seconds"] == pytest.approx(150.0)
        report, seen = self._run(
            tmp_path, monkeypatch, floor=0.4, highwater=0.77, reads=[0.3, 0.6],
        )
        assert seen["max_seconds"] is None

    def test_lifted_floor_caps_the_lr_at_the_probe_validated_arm(
        self, tmp_path, monkeypatch,
    ):
        tuner = _lif_tuner(tmp_path)
        tuner.pipeline_lr = 0.003  # the tier-0 pipeline lr; dip ~1.6k steps
        tuner.pipeline.config["endpoint_target_floor"] = 0.9
        _prepare_endpoint_scaffold(tuner)
        dhat_highwater.observe(tuner.pipeline, 0.5)
        reads = iter([0.3, 0.6])
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
        )
        seen = {}

        def fake_train(trainer, lr, target, *, max_steps, **kwargs):
            seen["lr"] = lr
            return 0.6, 9

        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(fake_train),
        )
        try:
            run_endpoint_recovery(tuner, base_steps=10)
        finally:
            tuner.close()
        assert seen["lr"] == pytest.approx(2e-3)

    def test_gentler_pipeline_lr_is_never_raised_to_the_floor_lr(
        self, tmp_path, monkeypatch,
    ):
        report, seen = self._run(
            tmp_path, monkeypatch, floor=0.9, highwater=0.5, reads=[0.3, 0.6],
        )
        assert seen["lr"] == pytest.approx(0.001)

    def test_entry_at_or_above_floor_does_not_engage(self, tmp_path, monkeypatch):
        report, seen = self._run(
            tmp_path, monkeypatch, floor=0.9, highwater=0.5, reads=[0.95],
        )
        assert report.engaged is False
        assert report.exit == pytest.approx(0.95)
        assert "target" not in seen

    def test_m_safety_exit_never_below_entry_under_the_floor(
        self, tmp_path, monkeypatch,
    ):
        # The floor changes how far ABOVE entry the stage may chase, never the
        # never-below-entry contract: a wrecked floor-chase rolls back to entry.
        tuner = _lif_tuner(tmp_path)
        tuner.pipeline.config["endpoint_target_floor"] = 0.98
        _prepare_endpoint_scaffold(tuner)
        dhat_highwater.observe(tuner.pipeline, 0.5)
        pre_sd = {k: v.clone() for k, v in tuner.model.state_dict().items()}
        reads = iter([0.6, 0.2])

        def corrupting_train(trainer, lr, target, **kwargs):
            with torch.no_grad():
                for p in trainer.model.parameters():
                    p.add_(1.0)
            return 0.2, 5

        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
        )
        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(corrupting_train),
        )
        try:
            report = run_endpoint_recovery(tuner, base_steps=10)
        finally:
            tuner.close()
        assert report.rolled_back is True
        assert report.exit == pytest.approx(0.6)
        post_sd = tuner.model.state_dict()
        for key in pre_sd:
            assert torch.equal(pre_sd[key], post_sd[key]), key

    def test_reached_false_is_a_legal_floor_outcome(self, tmp_path, monkeypatch):
        # The floor target ceases to be a reachability certificate.
        report, _ = self._run(
            tmp_path, monkeypatch, floor=0.98, highwater=0.5, reads=[0.3, 0.6],
        )
        assert report.reached is False
        assert report.exit == pytest.approx(0.6)


class TestExplicitTargetFloorAndWallCap:
    """[5u generalized] a call site may pass an explicit floor (the WQ endpoint
    scopes the well-conditioned floor to the final composition alone);
    ``target_floor=None`` keeps the config-read bit-parity every-endpoint path.
    The floor-lifted wall cap is config-overridable per-cell headroom."""

    def _drive(self, tmp_path, monkeypatch, *, target_floor, config_floor=None,
               wall_s=None, highwater=0.5, reads=(0.3, 0.6), base_steps=100):
        tuner = _lif_tuner(tmp_path)
        if config_floor is not None:
            tuner.pipeline.config["endpoint_target_floor"] = config_floor
        if wall_s is not None:
            tuner.pipeline.config["endpoint_floor_wall_s"] = wall_s
        _prepare_endpoint_scaffold(tuner)
        dhat_highwater.observe(tuner.pipeline, highwater)
        read_iter = iter(reads)
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(read_iter),
        )
        seen = {}

        def fake_train(trainer, lr, target, *, max_steps, **kwargs):
            seen.update(kwargs, target=target, lr=lr, max_steps=max_steps)
            return reads[-1], 9

        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(fake_train),
        )
        try:
            report = run_endpoint_recovery(
                tuner, base_steps=base_steps, target_floor=target_floor,
            )
        finally:
            tuner.close()
        return report, seen

    def test_explicit_floor_overrides_absent_config(self, tmp_path, monkeypatch):
        report, seen = self._drive(tmp_path, monkeypatch, target_floor=0.9)
        assert seen["target"] == pytest.approx(0.9)
        assert report.floor_lifted is True
        assert report.target_floor == pytest.approx(0.9)

    def test_explicit_floor_overrides_a_lower_config_floor(self, tmp_path, monkeypatch):
        report, seen = self._drive(
            tmp_path, monkeypatch, target_floor=0.9, config_floor=0.4,
        )
        assert seen["target"] == pytest.approx(0.9)

    def test_none_target_floor_reads_config(self, tmp_path, monkeypatch):
        report, seen = self._drive(
            tmp_path, monkeypatch, target_floor=None, config_floor=0.9,
        )
        assert seen["target"] == pytest.approx(0.9)
        assert report.floor_lifted is True

    def test_zero_explicit_floor_does_not_lift(self, tmp_path, monkeypatch):
        report, seen = self._drive(
            tmp_path, monkeypatch, target_floor=0.0, highwater=0.77,
        )
        assert seen["target"] == pytest.approx(0.77)
        assert report.floor_lifted is False

    def test_wall_cap_defaults_to_policy(self, tmp_path, monkeypatch):
        _, seen = self._drive(tmp_path, monkeypatch, target_floor=0.9)
        assert seen["max_seconds"] == pytest.approx(150.0)

    def test_wall_cap_reads_config_override(self, tmp_path, monkeypatch):
        _, seen = self._drive(tmp_path, monkeypatch, target_floor=0.9, wall_s=90.0)
        assert seen["max_seconds"] == pytest.approx(90.0)


class TestNeverBelowEntry:
    def test_regression_restores_the_entry_state(self, tmp_path, monkeypatch, capsys):
        tuner = _lif_tuner(tmp_path)
        try:
            _prepare_endpoint_scaffold(tuner)
            dhat_highwater.observe(tuner.pipeline, 0.9)
            pre_sd = {k: v.clone() for k, v in tuner.model.state_dict().items()}
            reads = iter([0.6, 0.2])  # entry 0.6, post-training crater 0.2

            def corrupting_train(trainer, lr, target, **kwargs):
                with torch.no_grad():
                    for p in trainer.model.parameters():
                        p.add_(1.0)
                return 0.2, 5

            monkeypatch.setattr(
                endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
            )
            monkeypatch.setattr(
                RecoveryEngine, "train_to_target", staticmethod(corrupting_train),
            )
            report = run_endpoint_recovery(tuner, base_steps=10)
            assert report.rolled_back is True
            assert report.exit == pytest.approx(0.6), "exit reads the restored entry"
            post_sd = tuner.model.state_dict()
            for key in pre_sd:
                assert torch.equal(pre_sd[key], post_sd[key]), key
        finally:
            tuner.close()
        line = _endpoint_lines(capsys.readouterr().out)[0]
        assert "rolled_back=True" in line


class TestRealTrainingSmoke:
    def test_endpoint_stage_trains_and_reports(self, tmp_path, capsys):
        # No stubs: the real trainer loop runs a few bounded steps on the tiny
        # fixture and the stage ends at-or-above entry by construction.
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path)
        try:
            _prepare_endpoint_scaffold(tuner)
            dhat_highwater.observe(tuner.pipeline, 0.99)
            report = run_endpoint_recovery(tuner, base_steps=2)
            assert isinstance(report, EndpointRecoveryReport)
            assert report.engaged is True
            assert report.budget_steps == 2 + freed_ladder_steps(tuner)
            assert report.exit >= report.entry - 1e-9
        finally:
            tuner.close()
        assert len(_endpoint_lines(capsys.readouterr().out)) == 1


class TestCallSites:
    def _capture(self, monkeypatch, module):
        calls = []
        monkeypatch.setattr(
            module, "run_endpoint_recovery",
            lambda tuner, *, base_steps: calls.append(base_steps),
        )
        return calls

    def test_lif_hook_runs_the_endpoint_stage(self, tmp_path, monkeypatch):
        import mimarsinan.tuning.tuners.lif_adaptation_tuner as lif_mod

        calls = self._capture(monkeypatch, lif_mod)
        tuner = _lif_tuner(tmp_path, endpoint_steps=123)
        try:
            _prepare_endpoint_scaffold(tuner)
            tuner._post_stabilization_hook()
            assert calls == [123]
            assert not hasattr(tuner, "_fast_stabilize"), (
                "the endpoint stage REPLACES _fast_stabilize (one home)"
            )
        finally:
            tuner.close()

    def test_lif_controller_path_does_not_engage(self, tmp_path, monkeypatch):
        import mimarsinan.tuning.tuners.lif_adaptation_tuner as lif_mod

        calls = self._capture(monkeypatch, lif_mod)
        tuner = _lif_tuner(tmp_path, endpoint_steps=123)
        try:
            tuner._fixed_ladder_policy = False
            tuner._post_stabilization_hook()
            assert calls == []
        finally:
            tuner.close()

    def test_ttfs_hook_runs_the_endpoint_stage(self, tmp_path, monkeypatch):
        import mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner as ttfs_mod
        from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
            TTFSCycleAdaptationTuner,
        )

        calls = self._capture(monkeypatch, ttfs_mod)
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 16
        cfg["ttfs_genuine_blend_ramp"] = True
        cfg["ttfs_genuine_blend_fast"] = True
        cfg["endpoint_recovery_steps"] = 300
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=make_tiny_supermodel(), target_accuracy=0.5,
            lr=cfg["lr"], adaptation_manager=AdaptationManager(),
        )
        try:
            tuner._post_stabilization_hook()
            assert calls == [300]
        finally:
            tuner.close()

    def test_sync_aq_hook_runs_the_endpoint_stage_and_zeroes_stabilize(
        self, tmp_path, monkeypatch,
    ):
        import mimarsinan.tuning.tuners.activation_quantization_tuner as aq_mod
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        calls = self._capture(monkeypatch, aq_mod)
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "synchronized"
        cfg["activation_quantization"] = True
        cfg["sync_exact_qat"] = True
        cfg["endpoint_recovery_steps"] = 600
        cfg["optimization_driver"] = "fast"
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        manager = create_adaptation_manager_for_model(cfg, model)
        tuner = ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)
        try:
            assert tuner._stabilization_budget() == 0, (
                "the endpoint stage replaces the open-ended sync AQ stabilize"
            )
            tuner._post_stabilization_hook()
            assert calls == [600]
        finally:
            tuner.close()

    def test_non_sync_aq_keeps_the_legacy_stabilize_and_no_endpoint(
        self, tmp_path, monkeypatch,
    ):
        import mimarsinan.tuning.tuners.activation_quantization_tuner as aq_mod
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        calls = self._capture(monkeypatch, aq_mod)
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_quantized"
        cfg["activation_quantization"] = True
        cfg["optimization_driver"] = "fast"
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        manager = create_adaptation_manager_for_model(cfg, model)
        tuner = ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)
        try:
            assert tuner._stabilization_budget() == \
                4 * int(tuner._budget.max_training_steps)
            tuner._post_stabilization_hook()
            assert calls == []
        finally:
            tuner.close()
