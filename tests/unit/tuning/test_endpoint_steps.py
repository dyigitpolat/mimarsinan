"""The run-scoped endpoint STEP ledger: one training-step budget shared by every
armed endpoint stage.

Reproducibility contract: training budgets are denominated in optimizer steps,
never wall seconds — identical configs train identical step counts on any
hardware (same config + same seed => same step trajectory, modulo GPU
nondeterminism). Wall time is a pure measurement evaluated at harvest. The
ledger makes ``endpoint_floor_steps`` the RUN total: each armed stage gets what
remains, and an exhausted budget falls back to the default patience geometry
(cheap stop, keep-best intact).
"""

from __future__ import annotations

from conftest import MockPipeline
from mimarsinan.tuning.orchestration import dhat_highwater, endpoint_recovery
from mimarsinan.tuning.orchestration import endpoint_steps
from mimarsinan.tuning.orchestration.endpoint_recovery import run_endpoint_recovery
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine


class TestLedger:
    def test_fresh_pipeline_has_zero_consumed(self):
        pipeline = MockPipeline()
        assert endpoint_steps.consumed(pipeline) == 0

    def test_consume_accumulates(self):
        pipeline = MockPipeline()
        endpoint_steps.consume(pipeline, 4000)
        endpoint_steps.consume(pipeline, 2550)
        assert endpoint_steps.consumed(pipeline) == 6550

    def test_remaining_floors_at_zero(self):
        pipeline = MockPipeline()
        endpoint_steps.consume(pipeline, 20000)
        assert endpoint_steps.remaining(pipeline, 16000) == 0

    def test_negative_consumption_is_ignored(self):
        pipeline = MockPipeline()
        endpoint_steps.consume(pipeline, -5)
        assert endpoint_steps.consumed(pipeline) == 0


class TestEndpointStagesShareTheBudget:
    def _make_tuner(self, tmp_path):
        from conftest import default_config, make_tiny_supermodel
        from mimarsinan.tuning.orchestration.adaptation_manager import (
            AdaptationManager,
        )
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

        cfg = default_config()
        cfg["spiking_mode"] = "lif"
        cfg["firing_mode"] = "Default"
        cfg["thresholding_mode"] = "<"
        cfg["simulation_steps"] = 4
        cfg["lif_blend_fast"] = True
        cfg["lif_blend_fast_steps_per_rate"] = 2
        cfg["lif_blend_fast_rates"] = [0.5, 1.0]
        cfg["endpoint_recovery_steps"] = 0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        return LIFAdaptationTuner(
            pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
            adaptation_manager=AdaptationManager(),
        )

    def _drive(self, tuner, monkeypatch, *, base_steps=12000, steps_trained=None,
               entry=0.095, highwater=0.9937):
        tuner._phase_seconds = {}
        tuner._mbh_rung_index = -1
        tuner._mbh_gate_state = None
        tuner._rollback_tolerance = 0.0
        tuner._fast_optimizer_steps = len(tuner._fixed_ladder_rates) * 2
        dhat_highwater.observe(tuner.pipeline, highwater)
        reads = iter([entry, entry + 0.01])
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
        )
        seen = {}

        def fake_train(trainer, lr, target, *, max_steps, **kwargs):
            seen.update(kwargs, max_steps=max_steps, lr=lr)
            used = max_steps if steps_trained is None else steps_trained
            return entry + 0.01, used

        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(fake_train),
        )
        report = run_endpoint_recovery(
            tuner, base_steps=base_steps, target_floor=0.98,
        )
        return report, seen

    def test_successive_stages_get_the_remaining_step_budget(
        self, tmp_path, monkeypatch,
    ):
        tuner = self._make_tuner(tmp_path)
        try:
            # Stage 1: a 12k stage budget fits inside the 16k default ledger.
            report1, seen1 = self._drive(tuner, monkeypatch, base_steps=12000)
            assert seen1["max_steps"] == 12000
            assert seen1["min_steps"] == 12000
            assert endpoint_steps.consumed(tuner.pipeline) == 12000
            assert report1.budget_steps == 12000
            # Stage 2: only 4,000 ledger steps remain; the stage budget clamps.
            report2, seen2 = self._drive(tuner, monkeypatch, base_steps=12000)
            assert seen2["max_steps"] == 4000
            assert seen2["min_steps"] == 4000
            assert report2.budget_steps == 4000
            # Stage 3: ledger exhausted — fall back to the default patience
            # geometry (cheap stop) instead of burning another full budget.
            report3, seen3 = self._drive(
                tuner, monkeypatch, base_steps=12000, steps_trained=9,
            )
            assert seen3["min_steps"] == 0
            assert seen3["max_steps"] == 12000
            assert report3.entry_gap_armed is True
        finally:
            tuner.close()

    def test_config_total_still_overrides(self, tmp_path, monkeypatch):
        tuner = self._make_tuner(tmp_path)
        tuner.pipeline.config["endpoint_floor_steps"] = 20000
        try:
            _, seen1 = self._drive(tuner, monkeypatch, base_steps=16000)
            assert seen1["max_steps"] == 16000
            _, seen2 = self._drive(tuner, monkeypatch, base_steps=16000)
            assert seen2["max_steps"] == 4000
        finally:
            tuner.close()

    def test_consumption_is_the_steps_actually_trained(self, tmp_path, monkeypatch):
        tuner = self._make_tuner(tmp_path)
        try:
            # An early target-reach stop consumes only the steps it trained.
            self._drive(tuner, monkeypatch, base_steps=12000, steps_trained=926)
            assert endpoint_steps.consumed(tuner.pipeline) == 926
            _, seen2 = self._drive(tuner, monkeypatch, base_steps=16000)
            assert seen2["max_steps"] == 16000 - 926
        finally:
            tuner.close()

    def test_untrained_stage_consumes_nothing(self, tmp_path, monkeypatch):
        tuner = self._make_tuner(tmp_path)
        try:
            tuner._phase_seconds = {}
            tuner._mbh_rung_index = -1
            tuner._mbh_gate_state = None
            tuner._rollback_tolerance = 0.0
            tuner._fast_optimizer_steps = len(tuner._fixed_ladder_rates) * 2
            dhat_highwater.observe(tuner.pipeline, 0.9)
            monkeypatch.setattr(
                endpoint_recovery, "_fp32_deployed_read", lambda t: 0.95,
            )
            report = run_endpoint_recovery(tuner, base_steps=100)
            assert report.engaged is False
            assert endpoint_steps.consumed(tuner.pipeline) == 0
        finally:
            tuner.close()
