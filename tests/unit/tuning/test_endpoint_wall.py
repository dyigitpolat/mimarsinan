"""The run-scoped endpoint wall ledger: one wall budget shared by every armed
endpoint stage.

Item-1's entry-gap arming made every gapped endpoint train up to the wall cap;
with per-stage caps a mixer run burns cap x N stages (measured 380-540 s
artifact walls). The ledger makes ``endpoint_floor_wall_s`` the RUN total:
each armed stage gets what remains, and an exhausted budget falls back to the
default patience geometry (cheap stop, keep-best intact).
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline
from mimarsinan.tuning.orchestration import dhat_highwater, endpoint_recovery
from mimarsinan.tuning.orchestration import endpoint_wall
from mimarsinan.tuning.orchestration.endpoint_recovery import run_endpoint_recovery
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine


class TestLedger:
    def test_fresh_pipeline_has_zero_consumed(self):
        pipeline = MockPipeline()
        assert endpoint_wall.consumed(pipeline) == 0.0

    def test_consume_accumulates(self):
        pipeline = MockPipeline()
        endpoint_wall.consume(pipeline, 40.0)
        endpoint_wall.consume(pipeline, 25.5)
        assert endpoint_wall.consumed(pipeline) == pytest.approx(65.5)

    def test_remaining_floors_at_zero(self):
        pipeline = MockPipeline()
        endpoint_wall.consume(pipeline, 200.0)
        assert endpoint_wall.remaining(pipeline, 150.0) == 0.0

    def test_negative_consumption_is_ignored(self):
        pipeline = MockPipeline()
        endpoint_wall.consume(pipeline, -5.0)
        assert endpoint_wall.consumed(pipeline) == 0.0


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

    def _drive(self, tuner, monkeypatch, *, clock, elapsed_per_call,
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
            clock[0] += elapsed_per_call
            return entry + 0.01, 9

        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(fake_train),
        )
        monkeypatch.setattr(
            endpoint_recovery, "_monotonic", lambda: clock[0],
        )
        report = run_endpoint_recovery(tuner, base_steps=100, target_floor=0.98)
        return report, seen

    def test_successive_stages_get_the_remaining_budget(
        self, tmp_path, monkeypatch,
    ):
        tuner = self._make_tuner(tmp_path)
        clock = [1000.0]
        try:
            _, seen1 = self._drive(
                tuner, monkeypatch, clock=clock, elapsed_per_call=100.0,
            )
            assert seen1["max_seconds"] == pytest.approx(150.0)
            # Overruns past the cap (checkpoint-granularity) are consumed too.
            _, seen2 = self._drive(
                tuner, monkeypatch, clock=clock, elapsed_per_call=60.0,
            )
            assert seen2["max_seconds"] == pytest.approx(50.0)
            report3, seen3 = self._drive(
                tuner, monkeypatch, clock=clock, elapsed_per_call=1.0,
            )
            # Budget exhausted: the stage falls back to the default patience
            # geometry (cheap stop) instead of burning another cap.
            assert seen3["max_seconds"] is None
            assert seen3["min_steps"] == 0
            assert report3.entry_gap_armed is True
        finally:
            tuner.close()

    def test_config_total_still_overrides(self, tmp_path, monkeypatch):
        tuner = self._make_tuner(tmp_path)
        tuner.pipeline.config["endpoint_floor_wall_s"] = 600.0
        clock = [0.0]
        try:
            _, seen1 = self._drive(
                tuner, monkeypatch, clock=clock, elapsed_per_call=250.0,
            )
            assert seen1["max_seconds"] == pytest.approx(600.0)
            _, seen2 = self._drive(
                tuner, monkeypatch, clock=clock, elapsed_per_call=10.0,
            )
            assert seen2["max_seconds"] == pytest.approx(350.0)
        finally:
            tuner.close()

    def test_untrained_stage_consumes_nothing(self, tmp_path, monkeypatch):
        tuner = self._make_tuner(tmp_path)
        clock = [0.0]
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
            assert endpoint_wall.consumed(tuner.pipeline) == 0.0
        finally:
            tuner.close()
