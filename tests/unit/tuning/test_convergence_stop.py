"""[C1] convergence-stop for funded endpoint/stabilize stages.

A ledger-funded (armed) endpoint stage no longer disarms keep-best patience by
pinning ``min_steps = budget``: the stage first covers the measured lr dip
(min-cover = max(absolute ~2k cover, fraction of budget) — absolute at small
residual budgets), then stops on a budget-scaled patience. Budgets become true
CEILINGS, never mandatory burns (measured: 1.5-2.1 ks/campaign of zero-gain
burn on flat armed floors, and tier1/2 run the same geometry un-capped). The
legacy rate-1.0 stabilize seam gets the identical geometry, and every engaged
endpoint stage appends its per-check (step, progress_acc, best_acc) trajectory
to the ``mbh_endpoint`` event payload.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn as nn

from conftest import (
    MockDataProviderFactory,
    MockPipeline,
    default_config,
    make_tiny_supermodel,
)
from mimarsinan.tuning.orchestration import dhat_highwater
from mimarsinan.tuning.orchestration.frontier import endpoint_recovery
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.frontier.endpoint_recovery import (
    run_endpoint_recovery,
)
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
    SmoothAdaptationTuner,
)
from mimarsinan.tuning.orchestration.tuner_base import _RECOVERY_PATIENCE
from mimarsinan.tuning.orchestration.tuning_policy import (
    TUNING_POLICY,
    armed_endpoint_check_interval,
    armed_endpoint_effective_check_interval,
    endpoint_convergence_geometry,
)


# ── shared fixtures ──────────────────────────────────────────────────────────


def _lif_tuner(tmp_path, *, steps_per_rate=2, rates=(0.5, 1.0)):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = 4
    cfg["lif_blend_fast"] = True
    cfg["lif_blend_fast_steps_per_rate"] = steps_per_rate
    cfg["lif_blend_fast_rates"] = list(rates)
    cfg["endpoint_recovery_steps"] = 0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    return LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )


def _prepare_endpoint_scaffold(tuner):
    tuner._phase_seconds = {}
    tuner._mbh_rung_index = -1
    tuner._mbh_gate_state = None
    tuner._rollback_tolerance = 0.0
    tuner._fast_optimizer_steps = len(tuner._fixed_ladder_rates) * 2


def _step_trainer():
    """A real BasicTrainer on the tiny fixture (the loop is the unit under test)."""
    from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
    from mimarsinan.model_training.basic_trainer import BasicTrainer

    dl_factory = DataLoaderFactory(MockDataProviderFactory(), num_workers=0)
    model = make_tiny_supermodel()
    loss = nn.CrossEntropyLoss()
    return BasicTrainer(model, "cpu", dl_factory, lambda m, x, y: loss(m(x), y))


class _EventReporter:
    def __init__(self):
        self.events = []

    def report(self, *args, **kwargs):
        pass

    def console_log(self, *args, **kwargs):
        pass

    def event(self, kind, payload):
        self.events.append((kind, payload))

    def finish(self):
        pass


# ── the geometry SSOT ────────────────────────────────────────────────────────


class TestConvergenceGeometry:
    def test_policy_pins_the_calibrated_constants(self):
        assert TUNING_POLICY.endpoint_floor_min_cover_steps == 2000
        assert TUNING_POLICY.endpoint_floor_patience_fraction == 0.25

    def test_absolute_cover_wins_at_small_budgets(self):
        # A 600-step residual budget keeps the ABSOLUTE lr-dip cover: min-cover
        # above the ceiling means small budgets burn fully (never a premature
        # stop inside the dip).
        geometry = endpoint_convergence_geometry(600, 40)
        assert geometry.min_steps == 2000

    def test_fractional_cover_wins_at_large_budgets(self):
        geometry = endpoint_convergence_geometry(16000, 40)
        assert geometry.min_steps == 4000

    def test_patience_scales_with_the_budget_instead_of_disarmed(self):
        geometry = endpoint_convergence_geometry(16000, 40)
        assert geometry.patience == math.ceil(0.25 * 16000 / 40)

    def test_patience_is_at_least_one(self):
        assert endpoint_convergence_geometry(1, 100).patience == 1

    def test_geometry_reads_the_frozen_policy(self, monkeypatch):
        import mimarsinan.tuning.orchestration.tuning_policy as tp

        monkeypatch.setattr(
            tp, "TUNING_POLICY",
            dataclasses.replace(
                tp.TUNING_POLICY,
                endpoint_floor_min_cover_steps=50,
                endpoint_floor_patience_fraction=0.5,
            ),
        )
        geometry = tp.endpoint_convergence_geometry(100, 10)
        assert geometry.min_steps == 50
        assert geometry.patience == 5




def _widened_policy(monkeypatch):
    """Pin the widened-cadence MECHANISM (default is dense since the P4
    revert); mechanism tests exercise it via an explicit multiplier."""
    import mimarsinan.tuning.orchestration.tuning_policy as tp

    monkeypatch.setattr(tp, "TUNING_POLICY", dataclasses.replace(
        tp.TUNING_POLICY, endpoint_floor_eval_interval_multiplier=3))


class TestArmedEvalCadence:
    """[P4 part 2] the armed (ledger-funded) endpoint leg widens its keep-best
    check interval by a frozen policy multiplier, so armed stages spend
    <=~15% of wall on evals. Non-armed stages keep the exact
    ``_RECOVERY_PATIENCE x check_interval`` stagnation economics
    (mbh_analytical_ttfs_stagnation), and the [C1] geometry composes at the
    WIDENED interval: patience counts CHECKS, so the multiplier must never
    silently multiply the patience window's step budget."""

    def test_policy_pins_the_eval_cadence_multiplier(self):
        # [P4 reverted 2026-07-13] widened cadence measurably missed peaks at
        # any budget (t0_22: dense reached 0.9913, widened exit==entry).
        assert TUNING_POLICY.endpoint_floor_eval_interval_multiplier == 1

    def test_armed_cadence_is_the_multiplied_interval(self):
        assert armed_endpoint_check_interval(40) == 40
        assert armed_endpoint_check_interval(1) == 1

    def test_degenerate_intervals_are_floored_before_multiplying(self):
        assert armed_endpoint_check_interval(0) == 1

    def test_cadence_reads_the_frozen_policy(self, monkeypatch):
        import mimarsinan.tuning.orchestration.tuning_policy as tp

        monkeypatch.setattr(
            tp, "TUNING_POLICY",
            dataclasses.replace(
                tp.TUNING_POLICY, endpoint_floor_eval_interval_multiplier=5,
            ),
        )
        assert tp.armed_endpoint_check_interval(40) == 200

    def test_widened_cadence_never_multiplies_the_patience_step_window(self):
        # [C1 composition] patience counts CHECKS: the geometry computed at the
        # widened interval keeps the patience window ~fraction x budget in
        # STEPS (ceil slack of at most one widened interval), never 3x it.
        base_interval = 40
        widened = armed_endpoint_check_interval(base_interval)
        base = endpoint_convergence_geometry(16000, base_interval)
        wide = endpoint_convergence_geometry(16000, widened)
        base_window = base.patience * base_interval
        wide_window = wide.patience * widened
        assert base_window <= wide_window <= base_window + widened
        assert wide_window < 3 * base_window

    # [P4 trajectory-sensitivity fix] the lossless-wave v6 regressions (t0_22
    # sync lenet 0.9908->0.9850, t01_21 lif mixer wb8 0.9531->0.9461): with
    # few checks the coarser keep-best sampling misses the trajectory peak, so
    # a funded budget below the [C1] min-cover keeps the DENSE base cadence;
    # at/above the cover the [P4] multiplier applies.

    def test_small_funded_budget_keeps_the_dense_cadence(self):
        assert armed_endpoint_effective_check_interval(600, 40) == 40
        assert armed_endpoint_effective_check_interval(1999, 40) == 40

    def test_funded_budget_at_or_above_the_cover_widens(self, monkeypatch):
        _widened_policy(monkeypatch)
        assert armed_endpoint_effective_check_interval(2000, 40) == 120
        assert armed_endpoint_effective_check_interval(16000, 40) == 120

    def test_dense_threshold_is_the_c1_min_cover_ssot(self):
        cover = TUNING_POLICY.endpoint_floor_min_cover_steps
        interval = 40
        assert armed_endpoint_effective_check_interval(
            cover - 1, interval,
        ) == interval
        assert armed_endpoint_effective_check_interval(
            cover, interval,
        ) == armed_endpoint_check_interval(interval)

    def test_effective_cadence_reads_the_frozen_policy(self, monkeypatch):
        import mimarsinan.tuning.orchestration.tuning_policy as tp

        monkeypatch.setattr(
            tp, "TUNING_POLICY",
            dataclasses.replace(
                tp.TUNING_POLICY,
                endpoint_floor_min_cover_steps=50,
                endpoint_floor_eval_interval_multiplier=5,
            ),
        )
        assert tp.armed_endpoint_effective_check_interval(49, 40) == 40
        assert tp.armed_endpoint_effective_check_interval(50, 40) == 200

    def test_degenerate_intervals_are_floored_at_both_cadences(self, monkeypatch):
        _widened_policy(monkeypatch)
        assert armed_endpoint_effective_check_interval(100, 0) == 1
        assert armed_endpoint_effective_check_interval(5000, 0) == 3

    def test_c1_patience_window_invariant_holds_at_both_cadences(self):
        # [C1 composition] at EITHER effective cadence the patience window
        # stays ~fraction x budget in steps (ceil slack of one interval), and
        # min-cover keeps its absolute floor — the dense fallback never
        # weakens the geometry.
        fraction = TUNING_POLICY.endpoint_floor_patience_fraction
        for budget in (600, 1999, 2000, 16000):
            effective = armed_endpoint_effective_check_interval(budget, 40)
            geometry = endpoint_convergence_geometry(budget, effective)
            window = geometry.patience * effective
            assert fraction * budget <= window <= fraction * budget + effective
            assert geometry.min_steps >= (
                TUNING_POLICY.endpoint_floor_min_cover_steps
            )

    def test_dense_cadence_samples_the_trajectory_more_densely(self, monkeypatch):
        _widened_policy(monkeypatch)
        # The fix's mechanism: a sub-cover budget affords 3x the keep-best
        # reads of the widened cadence, so a between-checks peak stays visible.
        budget, base_interval = 600, 40
        effective = armed_endpoint_effective_check_interval(budget, base_interval)
        widened = armed_endpoint_check_interval(base_interval)
        assert budget // effective == 3 * (budget // widened)


# ── the keep-best loop honors the geometry (step-deterministic) ─────────────


class TestConvergenceStopLoop:
    def _flat_run(self, *, max_steps=400, check_interval=2, patience=5,
                  min_steps=30):
        trainer = _step_trainer()
        trainer.validate_n_batches = lambda n: 0.5
        _, steps = trainer.train_steps_until_target(
            lr=1e-3, max_steps=max_steps, target_accuracy=1.0,
            validation_n_batches=1, check_interval=check_interval,
            patience=patience, min_steps=min_steps, min_improvement=1e-3,
            cosine_decay=True, return_steps=True, final_validation=False,
        )
        return trainer, steps

    def test_flat_trace_stops_at_the_min_cover_and_restores_entry(self):
        # Stale checks accrue inside the cover; the stop lands on the first
        # check at-or-past min_steps, and keep-best restores the entry state.
        trainer = _step_trainer()
        pre_sd = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.validate_n_batches = lambda n: 0.5
        _, steps = trainer.train_steps_until_target(
            lr=1e-3, max_steps=400, target_accuracy=1.0,
            validation_n_batches=1, check_interval=2, patience=5,
            min_steps=30, min_improvement=1e-3, cosine_decay=True,
            return_steps=True, final_validation=False,
        )
        assert steps == 30
        post_sd = trainer.model.state_dict()
        for key in pre_sd:
            assert torch.equal(pre_sd[key], post_sd[key]), key

    def test_climbing_trace_runs_the_full_budget(self):
        trainer = _step_trainer()
        calls = [0]

        def climbing(n):
            calls[0] += 1
            return 0.1 + 0.01 * calls[0]

        trainer.validate_n_batches = climbing
        _, steps = trainer.train_steps_until_target(
            lr=1e-3, max_steps=60, target_accuracy=1.0,
            validation_n_batches=1, check_interval=2, patience=5,
            min_steps=30, min_improvement=1e-3, cosine_decay=True,
            return_steps=True, final_validation=False,
        )
        assert steps == 60

    def test_stop_is_step_deterministic(self):
        _, steps_a = self._flat_run()
        _, steps_b = self._flat_run()
        assert steps_a == steps_b == 30

    def test_non_armed_stagnation_stops_at_exactly_patience_times_interval(self):
        # The mbh_analytical_ttfs_stagnation economics: a stagnating NON-armed
        # endpoint (min_steps=0, pre-floor patience) stops at exactly
        # _RECOVERY_PATIENCE x check_interval steps — the [P4] armed cadence
        # multiplier must never touch this geometry.
        trainer = _step_trainer()
        trainer.validate_n_batches = lambda n: 0.5
        _, steps = trainer.train_steps_until_target(
            lr=1e-3, max_steps=400, target_accuracy=1.0,
            validation_n_batches=1, check_interval=4,
            patience=_RECOVERY_PATIENCE, min_steps=0, min_improvement=1e-3,
            cosine_decay=True, return_steps=True, final_validation=False,
        )
        assert steps == _RECOVERY_PATIENCE * 4

    def test_widened_cadence_runs_the_full_budget_with_fewer_reads(self):
        # [P4] an armed climbing leg at the widened interval still burns its
        # full step budget, with 1 entry read + floor(budget / interval)
        # keep-best reads (final_validation off).
        trainer = _step_trainer()
        calls = [0]

        def climbing(n):
            calls[0] += 1
            return 0.1 + 0.01 * calls[0]

        trainer.validate_n_batches = climbing
        _, steps = trainer.train_steps_until_target(
            lr=1e-3, max_steps=60, target_accuracy=1.0,
            validation_n_batches=1, check_interval=6, patience=5,
            min_steps=0, min_improvement=1e-3, cosine_decay=True,
            return_steps=True, final_validation=False,
        )
        assert steps == 60
        assert calls[0] == 1 + 60 // 6

    def test_widened_cadence_keeps_the_keep_best_entry_restore(self):
        trainer = _step_trainer()
        pre_sd = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.validate_n_batches = lambda n: 0.5
        trainer.train_steps_until_target(
            lr=1e-3, max_steps=60, target_accuracy=1.0,
            validation_n_batches=1, check_interval=6, patience=2,
            min_steps=0, min_improvement=1e-3, cosine_decay=True,
            return_steps=True, final_validation=False,
        )
        post_sd = trainer.model.state_dict()
        for key in pre_sd:
            assert torch.equal(pre_sd[key], post_sd[key]), key

    def test_cosine_horizon_is_bound_to_the_leg_budget(self):
        # [WS-X audit] the cosine schedule spans exactly the leg's max_steps
        # (the ledger-clamped budget at the endpoint seam), never a longer
        # grant that would stretch the climb past the fundable steps.
        trainer = _step_trainer()
        optimizer, scheduler, _ = trainer._get_optimizer_and_scheduler_steps(
            1e-3, 500,
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 500
        del optimizer, scheduler

    def test_composition_sub_se_unreachable_target_is_patience_stopped(
        self, monkeypatch,
    ):
        # [C1 x C2] the arm-when-engaged worst case: a sub-SE gap arms a big
        # funded budget against an unreachable target; the convergence stop
        # ends the burn at ~max(min-cover, fraction x budget), never the full
        # ledger. (Cover scaled down so the loop stays CPU-cheap.)
        import mimarsinan.tuning.orchestration.tuning_policy as tp

        monkeypatch.setattr(
            tp, "TUNING_POLICY",
            dataclasses.replace(tp.TUNING_POLICY, endpoint_floor_min_cover_steps=20),
        )
        geometry = tp.endpoint_convergence_geometry(400, 2)
        assert geometry.min_steps == 100
        trainer = _step_trainer()
        trainer.validate_n_batches = lambda n: 0.5
        _, steps = trainer.train_steps_until_target(
            lr=1e-3, max_steps=400, target_accuracy=0.5 + 1e-4,
            validation_n_batches=1, check_interval=2,
            patience=geometry.patience, min_steps=geometry.min_steps,
            min_improvement=1e-3, cosine_decay=True, return_steps=True,
            final_validation=False,
        )
        assert steps == geometry.min_steps
        assert steps < 400


# ── the v6 regression anatomy: peak capture needs the dense cadence ─────────


class TestTrajectoryPeakCapture:
    """[P4 trajectory-sensitivity fix] synthetic v6 trace: the keep-best peak
    sits on the dense check grid BETWEEN two coarse checks; the dense cadence
    a sub-cover budget now keeps captures it, while the widened cadence reads
    straight past it (the measured t0_22 / t01_21 regression mechanism)."""

    _BUDGET = 12
    _PEAK_STEP = 4  # on the dense grid (interval 2), off the widened grid (6)

    def _best_captured(self, check_interval):
        trainer = _step_trainer()
        calls = [0]

        def trace(n):
            step = calls[0] * check_interval  # entry read, then every check
            calls[0] += 1
            return 0.98 if step == self._PEAK_STEP else 0.5

        trainer.validate_n_batches = trace
        best_seen = [0.0]

        def on_check(step, acc, best_acc, entry_acc):
            best_seen[0] = best_acc
            return False

        trainer.train_steps_until_target(
            lr=1e-3, max_steps=self._BUDGET, target_accuracy=1.0,
            validation_n_batches=1, check_interval=check_interval,
            patience=99, min_steps=self._BUDGET, min_improvement=1e-3,
            cosine_decay=True, return_steps=True, final_validation=False,
            on_check=on_check,
        )
        return best_seen[0]

    def test_dense_cadence_captures_the_between_checks_peak(self):
        base_interval = 2
        effective = armed_endpoint_effective_check_interval(
            self._BUDGET, base_interval,
        )
        assert effective == base_interval, "sub-cover budget keeps dense cadence"
        assert self._best_captured(effective) == pytest.approx(0.98)

    def test_widened_cadence_misses_the_same_peak(self, monkeypatch):
        _widened_policy(monkeypatch)
        # The regression this fix removes for small budgets: the peak check
        # never happens at the coarse cadence, so keep-best holds the entry.
        widened = armed_endpoint_check_interval(2)
        assert self._PEAK_STEP % widened != 0, "fixture: peak off the wide grid"
        assert self._best_captured(widened) == pytest.approx(0.5)


# ── the armed endpoint stage wires the geometry ─────────────────────────────


class TestArmedStageGeometry:
    def _drive(self, tmp_path, monkeypatch, *, base_steps, floor=0.9,
               highwater=0.5, reads=(0.3, 0.6)):
        tuner = _lif_tuner(tmp_path)
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
                tuner, base_steps=base_steps, target_floor=floor,
            )
        finally:
            interval = tuner._budget.check_interval
            tuner.close()
        return report, seen, interval

    def test_armed_min_steps_is_the_cover_not_the_budget(
        self, tmp_path, monkeypatch,
    ):
        report, seen, _ = self._drive(tmp_path, monkeypatch, base_steps=16000)
        assert seen["max_steps"] == 16000
        assert seen["min_steps"] == 4000
        assert report.armed is True

    def test_min_cover_stays_absolute_at_small_residual_budgets(
        self, tmp_path, monkeypatch,
    ):
        _, seen, _ = self._drive(tmp_path, monkeypatch, base_steps=300)
        assert seen["max_steps"] == 300
        assert seen["min_steps"] == 2000

    def test_armed_patience_is_budget_scaled_never_disarmed(
        self, tmp_path, monkeypatch,
    ):
        _, seen, interval = self._drive(tmp_path, monkeypatch, base_steps=16000)
        expected = endpoint_convergence_geometry(
            16000, armed_endpoint_check_interval(interval),
        )
        assert seen["patience"] == expected.patience
        assert seen["patience"] < 16000 // max(1, interval), (
            "patience must be able to fire inside the budget (true ceiling)"
        )

    def test_armed_stage_widens_the_keep_best_cadence(
        self, tmp_path, monkeypatch,
    ):
        _widened_policy(monkeypatch)
        # [P4 part 2] the ledger-funded leg checks at the multiplied interval;
        # the geometry is computed at the SAME widened interval so the patience
        # step-window stays ~fraction x budget.
        _, seen, interval = self._drive(tmp_path, monkeypatch, base_steps=16000)
        widened = armed_endpoint_check_interval(interval)
        assert seen["check_interval"] == widened == 3 * interval
        base_window = (
            endpoint_convergence_geometry(16000, interval).patience * interval
        )
        wide_window = seen["patience"] * seen["check_interval"]
        assert base_window <= wide_window <= base_window + widened

    def test_small_funded_budget_stage_keeps_the_dense_cadence(
        self, tmp_path, monkeypatch,
    ):
        # [v6 fix] the t0_22/t01_21 anatomy: a sub-cover funded budget keeps
        # the DENSE keep-best cadence, with the [C1] geometry composed at that
        # same base interval.
        _, seen, interval = self._drive(tmp_path, monkeypatch, base_steps=300)
        assert seen["check_interval"] == interval
        expected = endpoint_convergence_geometry(300, interval)
        assert seen["min_steps"] == expected.min_steps == 2000
        assert seen["patience"] == expected.patience

    def test_armed_stage_keeps_cosine_and_the_step_denomination(
        self, tmp_path, monkeypatch,
    ):
        _, seen, _ = self._drive(tmp_path, monkeypatch, base_steps=16000)
        assert seen["cosine_decay"] is True
        assert "max_seconds" not in seen


# ── the legacy stabilize seam gets the identical geometry ───────────────────


class _StabTuner(SmoothAdaptationTuner):
    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.train_calls = []

    def _update_and_evaluate(self, rate):
        return 0.9

    def _find_lr(self):
        return 0.001

    def _recovery_training_hooks(self, rate):
        return []


def _stab_tuner(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    tuner = _StabTuner(pipeline, make_tiny_supermodel(), 0.9, 0.001)

    def _fake_train(lr, max_steps, target, *args, **kwargs):
        tuner.train_calls.append(
            {"lr": lr, "max_steps": int(max_steps), "kwargs": kwargs}
        )

    tuner.trainer.train_steps_until_target = _fake_train
    tuner.trainer.validate_n_batches = lambda n: 0.9
    tuner._committed_rate = 1.0
    tuner._validation_baseline = 0.9
    tuner._pipeline_hard_floor = None
    tuner._rollback_tolerance = 0.05
    tuner._cached_lr = 0.0005
    return tuner


class TestStabilizeSeamGeometry:
    def test_stabilize_uses_the_convergence_stop_geometry(self, tmp_path):
        tuner = _stab_tuner(tmp_path)
        try:
            tuner._stabilize_at_full_rate()
            budget = 2 * tuner._budget.max_training_steps
            expected = endpoint_convergence_geometry(
                budget, tuner._budget.check_interval,
            )
            assert len(tuner.train_calls) == 1
            call = tuner.train_calls[0]
            assert call["max_steps"] == budget, "the budget stays the ceiling"
            assert call["kwargs"]["min_steps"] == expected.min_steps
            assert call["kwargs"]["patience"] == expected.patience
        finally:
            tuner.close()


# ── per-check trajectory telemetry on the mbh_endpoint event ────────────────


class TestTrajectoryTelemetry:
    def _endpoint_payloads(self, reporter):
        return [p for kind, p in reporter.events if kind == "mbh_endpoint"]

    def test_engaged_stage_appends_the_per_check_trajectory(self, tmp_path, monkeypatch):
        # C1 telemetry under a single leg: pin the C3 rescue off so a dead
        # trace cannot append a restarted second leg to the trajectory.
        monkeypatch.setattr(
            endpoint_recovery, "TUNING_POLICY",
            dataclasses.replace(
                endpoint_recovery.TUNING_POLICY,
                endpoint_floor_divergence_rescue=False,
            ),
        )
        torch.manual_seed(0)
        tuner = _lif_tuner(tmp_path)
        tuner.pipeline.reporter = _EventReporter()
        try:
            _prepare_endpoint_scaffold(tuner)
            tuner._fast_optimizer_steps = 0  # freed ladder steps fund checks
            dhat_highwater.observe(tuner.pipeline, 0.99)
            run_endpoint_recovery(tuner, base_steps=4)
            payload = self._endpoint_payloads(tuner.pipeline.reporter)[0]
            trajectory = payload["trajectory"]
            assert len(trajectory) >= 1
            steps = [entry[0] for entry in trajectory]
            assert steps == sorted(steps)
            for step, progress_acc, best_acc in trajectory:
                assert isinstance(step, int)
                assert 0.0 <= progress_acc <= 1.0
                assert 0.0 <= best_acc <= 1.0
        finally:
            tuner.close()

    def test_disengaged_stage_emits_an_empty_trajectory(
        self, tmp_path, monkeypatch,
    ):
        tuner = _lif_tuner(tmp_path)
        tuner.pipeline.reporter = _EventReporter()
        try:
            _prepare_endpoint_scaffold(tuner)
            dhat_highwater.observe(tuner.pipeline, 0.10)
            monkeypatch.setattr(
                endpoint_recovery, "_fp32_deployed_read", lambda t: 0.9,
            )
            report = run_endpoint_recovery(tuner, base_steps=100)
            assert report.engaged is False
            payload = self._endpoint_payloads(tuner.pipeline.reporter)[0]
            assert payload["trajectory"] == []
        finally:
            tuner.close()
