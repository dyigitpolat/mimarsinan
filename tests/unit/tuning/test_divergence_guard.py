"""[C3] divergence guard + LR-backoff rescue on the armed endpoint floor.

Default-off ``TuningPolicy`` flag (``endpoint_floor_divergence_rescue``): when
on, the armed leg is watched by a per-check dead-run predicate — (a) the best
read never beat entry+SE after 5 checks, OR (b) the current read sat below the
run's pipeline hard floor for 3 consecutive checks (``None`` disables (b)
only). A fired leg restores its live keep-best (the loop's entry-anchored
restore), rebuilds the optimizer, and restarts the remaining funded budget
once at lr*0.3 with warmup = max(1, 2% of remaining) and cosine decay.
Measured motivation: the identical-config 3.5pp coin-flip (t0_21 0.9316 vs
sibling t01_24 0.9671) and the dead-16k floors (t01_05). Guard-off is
byte-identical to the C1 armed geometry.
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
from mimarsinan.tuning.orchestration import dhat_highwater, endpoint_steps
from mimarsinan.tuning.orchestration.frontier import endpoint_recovery
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.frontier.divergence_guard import (
    CRATER_CHECKS,
    TAKEOFF_CHECKS,
    CouplingGuard,
    DivergenceGuard,
    rescue_plan,
)
from mimarsinan.tuning.orchestration.frontier.endpoint_recovery import (
    run_endpoint_recovery,
)
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
from mimarsinan.tuning.orchestration.tuner_base import _RECOVERY_PATIENCE
from mimarsinan.tuning.orchestration.tuning_policy import (
    TUNING_POLICY,
    armed_endpoint_check_interval,
    endpoint_convergence_geometry,
)


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


def _enable_rescue(monkeypatch):
    monkeypatch.setattr(
        endpoint_recovery, "TUNING_POLICY",
        dataclasses.replace(
            endpoint_recovery.TUNING_POLICY,
            endpoint_floor_divergence_rescue=True,
        ),
    )


def _disable_rescue(monkeypatch):
    monkeypatch.setattr(
        endpoint_recovery, "TUNING_POLICY",
        dataclasses.replace(
            endpoint_recovery.TUNING_POLICY,
            endpoint_floor_divergence_rescue=False,
        ),
    )


# ── the predicate ────────────────────────────────────────────────────────────


class TestDivergenceGuardPredicate:
    def test_never_took_off_fires_at_exactly_the_takeoff_window(self):
        guard = DivergenceGuard(accuracy_se=0.02, hard_floor=None)
        for check in range(1, TAKEOFF_CHECKS):
            assert guard(check * 10, 0.5, 0.5, 0.5) is False, check
        assert guard(TAKEOFF_CHECKS * 10, 0.5, 0.5, 0.5) is True
        assert guard.fired is True

    def test_takeoff_before_the_window_disarms_the_dead_run_disjunct(self):
        guard = DivergenceGuard(accuracy_se=0.02, hard_floor=None)
        assert guard(10, 0.53, 0.53, 0.5) is False  # best beat entry+SE
        for check in range(2, 12):
            assert guard(check * 10, 0.5, 0.53, 0.5) is False, check
        assert guard.fired is False

    def test_post_progress_crater_fires_despite_takeoff(self):
        guard = DivergenceGuard(accuracy_se=0.02, hard_floor=0.4)
        assert guard(10, 0.9, 0.9, 0.5) is False
        assert guard(20, 0.1, 0.9, 0.5) is False
        assert guard(30, 0.1, 0.9, 0.5) is False
        assert guard(40, 0.1, 0.9, 0.5) is True
        assert guard.fired is True

    def test_crater_streak_resets_on_a_read_above_the_floor(self):
        guard = DivergenceGuard(accuracy_se=0.02, hard_floor=0.4)
        assert guard(10, 0.9, 0.9, 0.5) is False
        assert guard(20, 0.1, 0.9, 0.5) is False
        assert guard(30, 0.1, 0.9, 0.5) is False
        assert guard(40, 0.45, 0.9, 0.5) is False  # streak resets
        assert guard(50, 0.1, 0.9, 0.5) is False
        assert guard(60, 0.1, 0.9, 0.5) is False
        assert guard.fired is False

    def test_none_hard_floor_disables_only_the_crater_disjunct(self):
        cratering = DivergenceGuard(accuracy_se=0.02, hard_floor=None)
        assert cratering(10, 0.9, 0.9, 0.5) is False  # takeoff
        for check in range(2, 10):
            assert cratering(check * 10, 0.1, 0.9, 0.5) is False, check
        assert cratering.fired is False
        dead = DivergenceGuard(accuracy_se=0.02, hard_floor=None)
        for check in range(1, TAKEOFF_CHECKS):
            assert dead(check * 10, 0.5, 0.5, 0.5) is False
        assert dead(TAKEOFF_CHECKS * 10, 0.5, 0.5, 0.5) is True

    def test_guard_is_one_shot(self):
        guard = DivergenceGuard(accuracy_se=0.02, hard_floor=None)
        for check in range(1, TAKEOFF_CHECKS + 1):
            guard(check * 10, 0.5, 0.5, 0.5)
        assert guard.fired is True
        assert guard(999, 0.99, 0.99, 0.5) is True

    def test_predicate_constants_are_the_measured_values(self):
        assert TAKEOFF_CHECKS == 5
        assert CRATER_CHECKS == 3


# ── the coupling guard (decoupled surrogate↔deployed legs stop) ─────────────


class _DeployedReads:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.values[min(self.calls - 1, len(self.values) - 1)]


class TestCouplingGuardPredicate:
    """[C1'] the leg trains the SURROGATE but the funded objective is the
    DEPLOYED read: when the surrogate gains >= SE over a cadence window while
    the deployed read gains < SE, further funding buys quality the deployed
    composition cannot express (measured: a ViT WQ leg burned 3.5 h at flat
    deployed 0.396 while its surrogate climbed 0.43->0.61)."""

    def test_decoupled_leg_fires_at_the_first_window(self):
        reads = _DeployedReads([0.396])
        guard = CouplingGuard(
            deployed_read=reads, entry_deployed=0.396,
            accuracy_se=0.01, cadence=3,
        )
        assert guard(10, 0.44, 0.44, 0.43) is False
        assert guard(20, 0.47, 0.47, 0.43) is False
        # third check = the window boundary: surrogate +0.08 >= SE, deployed +0.
        assert guard(30, 0.51, 0.51, 0.43) is True
        assert guard.fired is True
        assert reads.calls == 1

    def test_coupled_leg_never_fires(self):
        reads = _DeployedReads([0.5, 0.6, 0.7])
        guard = CouplingGuard(
            deployed_read=reads, entry_deployed=0.4,
            accuracy_se=0.01, cadence=2,
        )
        surrogate = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        for check, acc in enumerate(surrogate, start=1):
            assert guard(check * 10, acc, acc, 0.4) is False, check
        assert guard.fired is False

    def test_plateaued_surrogate_never_fires(self):
        # Both flat: the stall belongs to [C1] patience, not the coupling guard.
        reads = _DeployedReads([0.4, 0.4, 0.4])
        guard = CouplingGuard(
            deployed_read=reads, entry_deployed=0.4,
            accuracy_se=0.01, cadence=2,
        )
        for check in range(1, 9):
            assert guard(check * 10, 0.5, 0.5, 0.5) is False, check
        assert guard.fired is False

    def test_deployed_reads_only_at_the_cadence(self):
        reads = _DeployedReads([0.4, 0.4])
        guard = CouplingGuard(
            deployed_read=reads, entry_deployed=0.4,
            accuracy_se=0.5, cadence=4,
        )
        for check in range(1, 9):
            guard(check * 10, 0.41, 0.41, 0.4)
        assert reads.calls == 2  # checks 4 and 8 only

    def test_windows_compare_gains_since_the_previous_window(self):
        # Window 1: surrogate +0.08 but deployed follows (+0.08) -> silent.
        # Window 2: surrogate +0.08 more, deployed flat -> fires.
        reads = _DeployedReads([0.48, 0.48])
        guard = CouplingGuard(
            deployed_read=reads, entry_deployed=0.40,
            accuracy_se=0.01, cadence=1,
        )
        assert guard(10, 0.51, 0.51, 0.43) is False
        assert guard(20, 0.59, 0.59, 0.43) is True

    def test_guard_is_one_shot(self):
        reads = _DeployedReads([0.4])
        guard = CouplingGuard(
            deployed_read=reads, entry_deployed=0.4,
            accuracy_se=0.01, cadence=1,
        )
        assert guard(10, 0.5, 0.5, 0.4) is True
        assert guard(20, 0.5, 0.5, 0.4) is True
        assert reads.calls == 1

    def test_policy_arms_the_guard_by_default(self):
        assert TUNING_POLICY.endpoint_coupling_guard is True


class TestCouplingGuardInStage:
    _drive = None  # bound below to reuse TestArmedStageRescue._drive

    def test_decoupled_leg_stops_without_rescue(self, tmp_path, monkeypatch):
        # Surrogate takes off fast (divergence guard silent) while deployed
        # reads stay flat: the coupling guard fires and the stage must NOT
        # rescue-restart (retraining the surrogate cannot buy deployed gain
        # on a decoupled leg).
        _enable_rescue(monkeypatch)

        def decoupled_leg(kwargs, max_steps):
            on_check = kwargs["on_check"]
            for check in range(1, 500):
                acc = min(0.99, 0.43 + 0.05 * check)
                if on_check(check * 3, acc, acc, 0.43):
                    return acc, check * 3
            pytest.fail("the coupling guard must fire within the leg")

        report, scripted, _, _ = TestArmedStageRescue._drive(
            self, tmp_path, monkeypatch,
            legs=[decoupled_leg, _never_called_leg],
            reads=(0.396,) * 600, base_steps=400,
        )
        assert len(scripted.calls) == 1
        assert report.divergence_rescued is False
        assert report.decoupled is True
        assert report.steps_used < 400

    def test_coupling_flag_off_reads_deployed_only_at_entry_and_exit(
        self, tmp_path, monkeypatch,
    ):
        _disable_rescue(monkeypatch)
        monkeypatch.setattr(
            endpoint_recovery, "TUNING_POLICY",
            dataclasses.replace(
                endpoint_recovery.TUNING_POLICY,
                endpoint_floor_divergence_rescue=False,
                endpoint_coupling_guard=False,
            ),
        )
        calls = [0]

        def counting_read(t):
            calls[0] += 1
            return 0.396

        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", counting_read,
        )

        def climbing_leg(kwargs, max_steps):
            on_check = kwargs["on_check"]
            for check in range(1, 10):
                on_check(check * 42, 0.43 + 0.02 * check, 0.43 + 0.02 * check, 0.43)
            return 0.6, 210

        tuner = _lif_tuner(tmp_path)
        _prepare_endpoint_scaffold(tuner)
        dhat_highwater.observe(tuner.pipeline, 0.9937)
        scripted = _ScriptedTrain([climbing_leg])
        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(scripted),
        )
        try:
            report = run_endpoint_recovery(
                tuner, base_steps=12000, target_floor=0.98,
            )
        finally:
            tuner.close()
        assert calls[0] == 2  # entry + exit only: flag-off is byte-identical
        assert report.decoupled is False


class TestMinCoverKnob:
    """[C3'] endpoint_floor_min_cover_steps is config-exposed: the geometry and
    the effective cadence both honor the override; the default is unchanged."""

    def test_config_override_reaches_the_geometry(self, tmp_path, monkeypatch):
        _disable_rescue(monkeypatch)
        tuner = _lif_tuner(tmp_path)
        tuner.pipeline.config["endpoint_floor_min_cover_steps"] = 100
        _prepare_endpoint_scaffold(tuner)
        dhat_highwater.observe(tuner.pipeline, 0.9937)
        reads = iter([0.095, 0.5])
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
        )
        scripted = _ScriptedTrain([lambda kwargs, max_steps: (0.5, 9)])
        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(scripted),
        )
        try:
            run_endpoint_recovery(tuner, base_steps=300, target_floor=0.98)
        finally:
            tuner.close()
        seen = scripted.calls[0]
        # default min-cover would be max(2000, ceil(0.25*300)) = 2000;
        # the override yields max(100, 75) = 100.
        assert seen["min_steps"] == 100

    def test_geometry_helper_takes_the_override(self):
        default = endpoint_convergence_geometry(300, 38)
        assert default.min_steps == TUNING_POLICY.endpoint_floor_min_cover_steps
        overridden = endpoint_convergence_geometry(300, 38, min_cover_steps=100)
        assert overridden.min_steps == 100
        # The fractional term still wins when larger.
        big = endpoint_convergence_geometry(12000, 38, min_cover_steps=100)
        assert big.min_steps == math.ceil(0.25 * 12000)


# ── the rescue restart geometry ──────────────────────────────────────────────


class TestRescuePlan:
    def test_plan_backs_off_the_lr_and_ramps_a_warmup(self):
        plan = rescue_plan(11790, 2e-3)
        assert plan is not None
        assert plan.lr == pytest.approx(2e-3 * 0.3)
        assert plan.warmup_steps == max(1, math.ceil(0.02 * 11790))
        assert plan.train_steps == 11790 - plan.warmup_steps

    def test_warmup_never_drops_below_one_step(self):
        plan = rescue_plan(10, 1e-3)
        assert plan is not None
        assert plan.warmup_steps == 1
        assert plan.train_steps == 9

    def test_total_never_exceeds_the_remaining_budget(self):
        for remaining in (2, 3, 50, 999, 16000):
            plan = rescue_plan(remaining, 1e-3)
            assert plan is not None
            assert plan.warmup_steps + plan.train_steps <= remaining, remaining

    def test_too_small_a_remainder_yields_no_rescue(self):
        assert rescue_plan(1, 1e-3) is None
        assert rescue_plan(0, 1e-3) is None

    def test_policy_pins_the_rescue_constants(self):
        assert TUNING_POLICY.endpoint_floor_divergence_rescue is True
        assert TUNING_POLICY.endpoint_floor_rescue_lr_factor == 0.3
        assert TUNING_POLICY.endpoint_floor_rescue_warmup_fraction == 0.02


# ── the loop aborts on a fired guard and restores the live best ─────────────


class TestGuardInTheLoop:
    def _trainer(self):
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
        from mimarsinan.model_training.basic_trainer import BasicTrainer

        dl_factory = DataLoaderFactory(MockDataProviderFactory(), num_workers=0)
        model = make_tiny_supermodel()
        loss = nn.CrossEntropyLoss()
        return BasicTrainer(model, "cpu", dl_factory, lambda m, x, y: loss(m(x), y))

    def test_fired_guard_aborts_the_leg_and_restores_entry(self):
        trainer = self._trainer()
        pre_sd = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.validate_n_batches = lambda n: 0.5
        guard = DivergenceGuard(accuracy_se=0.01, hard_floor=None)
        _, steps = trainer.train_steps_until_target(
            lr=1e-3, max_steps=100, target_accuracy=1.0,
            validation_n_batches=1, check_interval=1, patience=99,
            min_steps=100, min_improvement=1e-3, cosine_decay=True,
            return_steps=True, final_validation=False, on_check=guard,
        )
        assert guard.fired is True
        assert steps == TAKEOFF_CHECKS
        post_sd = trainer.model.state_dict()
        for key in pre_sd:
            assert torch.equal(pre_sd[key], post_sd[key]), key

    def test_dormant_guard_leaves_the_loop_untouched(self):
        trainer = self._trainer()
        calls = [0]

        def climbing(n):
            calls[0] += 1
            return 0.1 + 0.01 * calls[0]

        trainer.validate_n_batches = climbing
        guard = DivergenceGuard(accuracy_se=0.001, hard_floor=None)
        _, steps = trainer.train_steps_until_target(
            lr=1e-3, max_steps=40, target_accuracy=1.0,
            validation_n_batches=1, check_interval=2, patience=5,
            min_steps=20, min_improvement=1e-3, cosine_decay=True,
            return_steps=True, final_validation=False, on_check=guard,
        )
        assert guard.fired is False
        assert steps == 40


# ── the armed endpoint stage wraps the leg and rescues once ─────────────────


class _ScriptedTrain:
    """Fake ``RecoveryEngine.train_to_target`` driving the on_check seam."""

    def __init__(self, legs):
        self.legs = list(legs)
        self.calls = []

    def __call__(self, trainer, lr, target, *, max_steps, **kwargs):
        self.calls.append({"lr": lr, "target": target, "max_steps": max_steps,
                           **kwargs})
        leg = self.legs[len(self.calls) - 1]
        return leg(kwargs, max_steps)


def _dead_leg(exit_acc=0.095, steps_used=210):
    def leg(kwargs, max_steps):
        on_check = kwargs["on_check"]
        for check in range(1, 10):
            if on_check(check * 42, exit_acc, exit_acc, exit_acc):
                break
        return exit_acc, steps_used

    return leg


def _never_called_leg(kwargs, max_steps):
    pytest.fail("no second training leg may run")


class TestArmedStageRescue:
    def _drive(self, tmp_path, monkeypatch, *, legs, base_steps=12000,
               reads=(0.095, 0.5), hard_floor=None, exhaust_ledger=False):
        tuner = _lif_tuner(tmp_path)
        _prepare_endpoint_scaffold(tuner)
        tuner._pipeline_hard_floor = hard_floor
        if exhaust_ledger:
            endpoint_steps.consume(
                tuner.pipeline, TUNING_POLICY.endpoint_floor_steps,
            )
        dhat_highwater.observe(tuner.pipeline, 0.9937)
        read_iter = iter(reads)
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(read_iter),
        )
        scripted = _ScriptedTrain(legs)
        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(scripted),
        )
        try:
            report = run_endpoint_recovery(
                tuner, base_steps=base_steps, target_floor=0.98,
            )
        finally:
            interval = tuner._budget.check_interval
            tuner.close()
        return report, scripted, interval, tuner

    def test_guard_off_is_byte_identical_one_leg_no_warmup(
        self, tmp_path, monkeypatch,
    ):
        _disable_rescue(monkeypatch)
        report, scripted, _, _ = self._drive(
            tmp_path, monkeypatch, legs=[_dead_leg(), _never_called_leg],
        )
        assert len(scripted.calls) == 1
        assert "warmup_steps" not in scripted.calls[0]
        assert report.divergence_rescued is False
        assert report.steps_used == 210

    def test_fired_guard_restarts_the_remaining_budget_backed_off(
        self, tmp_path, monkeypatch,
    ):
        _enable_rescue(monkeypatch)
        rescue_seen = {}

        def rescue_leg(kwargs, max_steps):
            rescue_seen.update(kwargs, max_steps=max_steps)
            return 0.5, max_steps

        report, scripted, interval, _ = self._drive(
            tmp_path, monkeypatch, legs=[_dead_leg(), rescue_leg],
        )
        assert len(scripted.calls) == 2
        first, second = scripted.calls
        remaining = 12000 - 210
        warmup = max(1, math.ceil(0.02 * remaining))
        assert second["lr"] == pytest.approx(first["lr"] * 0.3)
        assert second["warmup_steps"] == warmup
        assert second["max_steps"] == remaining - warmup
        assert second["cosine_decay"] is True
        # [P4] the rescue leg is armed-only, so it keeps the widened cadence.
        widened = armed_endpoint_check_interval(interval)
        assert second["check_interval"] == widened
        geometry = endpoint_convergence_geometry(remaining - warmup, widened)
        assert second["min_steps"] == geometry.min_steps
        assert second["patience"] == geometry.patience
        assert report.divergence_rescued is True
        assert report.steps_used == 210 + (remaining - warmup)

    def test_rescue_charges_the_ledger_with_both_legs(
        self, tmp_path, monkeypatch,
    ):
        _enable_rescue(monkeypatch)
        report, _, _, tuner = self._drive(
            tmp_path, monkeypatch,
            legs=[_dead_leg(), lambda kwargs, max_steps: (0.5, max_steps)],
        )
        assert endpoint_steps.consumed(tuner.pipeline) == report.steps_used

    def test_healthy_leg_never_rescues(self, tmp_path, monkeypatch):
        _enable_rescue(monkeypatch)

        def healthy_leg(kwargs, max_steps):
            on_check = kwargs["on_check"]
            for check in range(1, 10):
                acc = 0.095 + 0.1 * check
                assert on_check(check * 42, acc, acc, 0.095) is False
            return 0.9, max_steps

        report, scripted, _, _ = self._drive(
            tmp_path, monkeypatch, legs=[healthy_leg, _never_called_leg],
            reads=(0.095, 0.9),
        )
        assert len(scripted.calls) == 1
        assert report.divergence_rescued is False

    def test_hard_floor_threads_from_the_run_instance(
        self, tmp_path, monkeypatch,
    ):
        # A post-takeoff crater trace: fires only when the tuner carries a
        # pipeline hard floor; ``None`` disables the crater disjunct alone.
        _enable_rescue(monkeypatch)

        def crater_leg(kwargs, max_steps):
            on_check = kwargs["on_check"]
            fired = on_check(42, 0.9, 0.9, 0.5)
            for check in range(2, 5):
                fired = on_check(check * 42, 0.1, 0.9, 0.5)
                if fired:
                    break
            return 0.1, 168

        report, scripted, _, _ = self._drive(
            tmp_path, monkeypatch,
            legs=[crater_leg, lambda kwargs, max_steps: (0.5, max_steps)],
            hard_floor=0.4, reads=(0.5, 0.5),
        )
        assert len(scripted.calls) == 2
        assert report.divergence_rescued is True

        report, scripted, _, _ = self._drive(
            tmp_path, monkeypatch, legs=[crater_leg, _never_called_leg],
            hard_floor=None, reads=(0.5, 0.5),
        )
        assert len(scripted.calls) == 1
        assert report.divergence_rescued is False

    def test_unfunded_stage_never_carries_the_guard(
        self, tmp_path, monkeypatch,
    ):
        # Ledger exhausted: the stage falls back to the pre-floor patience
        # geometry and the guard does not arm even with the flag on.
        _enable_rescue(monkeypatch)
        report, scripted, _, _ = self._drive(
            tmp_path, monkeypatch, legs=[_dead_leg(), _never_called_leg],
            exhaust_ledger=True,
        )
        assert len(scripted.calls) == 1
        assert scripted.calls[0]["min_steps"] == 0
        assert scripted.calls[0]["patience"] == _RECOVERY_PATIENCE
        assert report.armed is False
        assert report.divergence_rescued is False

    def test_rescue_trajectory_spans_both_legs(self, tmp_path, monkeypatch):
        _enable_rescue(monkeypatch)

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

        tuner = _lif_tuner(tmp_path)
        tuner.pipeline.reporter = _EventReporter()
        _prepare_endpoint_scaffold(tuner)
        dhat_highwater.observe(tuner.pipeline, 0.9937)
        reads = iter([0.095, 0.5])
        monkeypatch.setattr(
            endpoint_recovery, "_fp32_deployed_read", lambda t: next(reads),
        )

        def rescue_leg(kwargs, max_steps):
            on_check = kwargs["on_check"]
            on_check(40, 0.4, 0.4, 0.095)
            on_check(80, 0.5, 0.5, 0.095)
            return 0.5, max_steps

        scripted = _ScriptedTrain([_dead_leg(), rescue_leg])
        monkeypatch.setattr(
            RecoveryEngine, "train_to_target", staticmethod(scripted),
        )
        try:
            run_endpoint_recovery(tuner, base_steps=12000, target_floor=0.98)
            payloads = [p for kind, p in tuner.pipeline.reporter.events
                        if kind == "mbh_endpoint"]
            trajectory = payloads[0]["trajectory"]
            assert len(trajectory) == TAKEOFF_CHECKS + 2
        finally:
            tuner.close()
