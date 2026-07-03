"""Four TuningPolicy-gated gradual-tuning levers (all default-off → byte-identical).

The gradual SmoothAdaptation ramp is KEPT; these levers only change recovery
quality, the stabilization tail, and the rollback gate's drift bound:

- CHANGE 1 ``rollback_ratchet`` + ``rollback_cumulative_bound``:
  a NON-STALLING ratchet. The per-step gate stays RELATIVE to ``pre_cycle_acc``
  (so the ramp keeps climbing — a naturally-declining surface still commits),
  but the CUMULATIVE drift below the best-committed high-water mark is capped at
  ``cumulative_bound``; the bound TIGHTENS as the best ratchets up.
- CHANGE 2 ``stabilization_bounded`` + ``stabilization_ratio``:
  ``train_steps_until_target`` reports the steps it ran; the tuner accumulates
  the gradual step total and runs a SINGLE bounded cosine-decay stabilization
  pass of ``ratio * gradual_steps`` steps (hard cutoff, no rounds/restarts).
- CHANGE 3 ``tight_plateau`` + ``recovery_check_divisor``: divide
  the recovery check interval so a plateau is detected in fewer steps.
- CHANGE 4 ``refind_lr_on_miss``: a MISS is the sole LR-cache invalidation
  trigger in the gradual cycle (behind the flag); a reached target never is.

The default-off regression at the end asserts the default TUNING_POLICY reproduces
the current per-cycle decisions and stabilization byte-identically.
"""

import pytest
import torch
import torch.nn as nn

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    make_scripted_run_tuner,
    default_config,
    override_tuning_policy,
    MockDataProviderFactory,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner
from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor


# ── shared scripted cycle tuner ──────────────────────────────────────────────


class _ScriptedCycleTuner(SmoothAdaptationTuner):
    """Drives ``_adaptation`` with a scripted ``[pre, post]`` validate sequence."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._committed_rate = 0.0
        self._validate_seq = []
        self._idx = 0
        self.find_lr_calls = 0
        self.trainer.train_steps_until_target = lambda *a, **k: None

    def _update_and_evaluate(self, rate):
        return 0.85  # non-catastrophic instant accuracy

    def _find_lr(self):
        self.find_lr_calls += 1
        return 0.001

    def _validate_n(self, _n):
        i = self._idx
        self._idx += 1
        return self._validate_seq[i] if i < len(self._validate_seq) else self._validate_seq[-1]

    def drive(self, rate, validate_seq):
        self._validate_seq = list(validate_seq)
        self._idx = 0
        self._last_post_acc = None  # force fresh pre_cycle_acc from the sequence
        self.trainer.validate_n_batches = self._validate_n
        return self._adaptation(rate)


def _make_cycle_tuner(tmp_path, monkeypatch=None, **policy_overrides):
    if policy_overrides:
        override_tuning_policy(monkeypatch, **policy_overrides)
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    tuner = _ScriptedCycleTuner(pipeline, make_tiny_supermodel(), 0.9, 0.001)
    tuner._rollback_tolerance = 0.05
    tuner._validation_baseline = 0.9
    return tuner


# ── CHANGE 1: non-stalling ratchet (relative gate + cumulative drift bound) ───


class TestNonStallingRatchet:
    def test_declining_surface_keeps_committing_does_not_stall(self, tmp_path, deterministic_rng, monkeypatch):
        """A surface that gives back a LITTLE each step still commits and the ramp
        progresses — the per-step gate is RELATIVE (no best-anchor stall)."""
        tuner = _make_cycle_tuner(
            tmp_path,
            monkeypatch,
            rollback_ratchet=True,
            rollback_cumulative_bound=0.05,
        )
        tuner._rollback_tolerance = 0.02
        tuner._validation_baseline = 0.80  # abs floor 0.76, well below the gate
        tuner._best_committed_acc = 0.90

        # Each step slips 0.01 < the 0.02 noise margin off the PREVIOUS, and the
        # cumulative drop from best 0.90 stays within the 0.05 bound. All commit.
        assert tuner.drive(0.3, validate_seq=[0.90, 0.89]) == 0.3
        assert tuner.drive(0.4, validate_seq=[0.89, 0.88]) == 0.4
        assert tuner.drive(0.5, validate_seq=[0.88, 0.87]) == 0.5
        assert tuner.drive(0.6, validate_seq=[0.87, 0.86]) == 0.6
        # best stays the high-water mark (commits never raised it here).
        assert tuner._best_committed_acc == pytest.approx(0.90)
        tuner.close()

    def test_drift_beyond_cumulative_bound_is_rolled_back(self, tmp_path, deterministic_rng, monkeypatch):
        """A step that would push post_acc MORE than cumulative_bound below the
        best is rolled back even though the per-step relative gate would admit it
        (the cumulative drift is capped — no accumulation to the floor)."""
        tuner = _make_cycle_tuner(
            tmp_path,
            monkeypatch,
            rollback_ratchet=True,
            rollback_cumulative_bound=0.05,
        )
        tuner._rollback_tolerance = 0.02
        tuner._validation_baseline = 0.80  # abs floor 0.76, below the gate
        tuner._best_committed_acc = 0.90

        # pre 0.86, post 0.84: relative gate (0.86-0.02=0.84) ADMITS, abs floor
        # 0.76 admits, but best-bound 0.90-0.05=0.85 → 0.84 < 0.85 → ROLLBACK.
        r = tuner.drive(0.5, validate_seq=[0.86, 0.84])
        assert r == tuner._committed_rate
        assert r != 0.5, "cumulative drift past the bound must roll back"
        tuner.close()

    def test_relative_gate_is_the_stricter_when_it_exceeds_the_bound(self, tmp_path, deterministic_rng, monkeypatch):
        """The threshold is the MAX of the three lower bounds, so a sharp single-
        step drop (relative gate strictest) still rolls back."""
        tuner = _make_cycle_tuner(
            tmp_path,
            monkeypatch,
            rollback_ratchet=True,
            rollback_cumulative_bound=0.30,  # loose cumulative bound
        )
        tuner._rollback_tolerance = 0.02
        tuner._validation_baseline = 0.50  # abs floor 0.475, far below
        tuner._best_committed_acc = 0.90

        # pre 0.90, post 0.80: relative gate 0.90-0.02=0.88 → 0.80 < 0.88 →
        # rollback (relative is stricter than best-bound 0.90-0.30=0.60).
        r = tuner.drive(0.5, validate_seq=[0.90, 0.80])
        assert r != 0.5, "a sharp single-step drop is caught by the relative gate"
        tuner.close()

    def test_new_high_tightens_the_cumulative_bound(self, tmp_path, deterministic_rng, monkeypatch):
        """A new committed high ratchets the best up, so the SAME post_acc that
        committed before the high now violates the tightened cumulative bound."""
        tuner = _make_cycle_tuner(
            tmp_path,
            monkeypatch,
            rollback_ratchet=True,
            rollback_cumulative_bound=0.05,
        )
        tuner._rollback_tolerance = 0.02
        tuner._validation_baseline = 0.50  # abs floor far below
        tuner._best_committed_acc = 0.85

        # Cycle 1: post 0.95 commits and ratchets the best up to 0.95.
        tuner.drive(0.3, validate_seq=[0.85, 0.95])
        assert tuner._best_committed_acc == pytest.approx(0.95)

        # Cycle 2: pre 0.95, post 0.89. relative gate 0.95-0.02=0.93 already
        # rejects, but even a within-noise slip would violate the tightened
        # cumulative bound 0.95-0.05=0.90 → 0.89 < 0.90.
        r2 = tuner.drive(0.4, validate_seq=[0.95, 0.89])
        assert r2 == 0.3, "ratcheted-up best tightens the cumulative bound"
        tuner.close()

    def test_flag_off_is_relative_to_previous_only(self, tmp_path, deterministic_rng, monkeypatch):
        """Flag off → the gate is relative-to-pre_cycle_acc exactly as today; the
        cumulative bound is NOT applied (small slips accumulate)."""
        tuner = _make_cycle_tuner(tmp_path, monkeypatch, rollback_ratchet=False)
        tuner._rollback_tolerance = 0.05
        tuner._validation_baseline = 0.80  # abs floor 0.76, below the gate
        # Slips of 0.02 each off the PREVIOUS accumulate well past best-margin.
        assert tuner.drive(0.3, validate_seq=[0.90, 0.88]) == 0.3
        assert tuner.drive(0.4, validate_seq=[0.88, 0.86]) == 0.4
        assert tuner.drive(0.5, validate_seq=[0.86, 0.84]) == 0.5
        tuner.close()

    def test_acceptance_sensor_non_stalling_ratchet_threshold_helper(self):
        """``ratchet_threshold`` is the MAX of relative, best-bound, abs floor."""
        # relative 0.86-0.02=0.84, best-bound 0.90-0.05=0.85, no floor → 0.85.
        assert AcceptanceSensor.ratchet_threshold(
            pre_cycle_acc=0.86, rollback_tolerance=0.02,
            best_committed_acc=0.90, cumulative_bound=0.05, absolute_floor=None,
        ) == pytest.approx(0.85)
        # relative 0.90-0.02=0.88 dominates the best-bound 0.90-0.30=0.60.
        assert AcceptanceSensor.ratchet_threshold(
            pre_cycle_acc=0.90, rollback_tolerance=0.02,
            best_committed_acc=0.90, cumulative_bound=0.30, absolute_floor=None,
        ) == pytest.approx(0.88)
        # abs floor 0.92 dominates both.
        assert AcceptanceSensor.ratchet_threshold(
            pre_cycle_acc=0.86, rollback_tolerance=0.02,
            best_committed_acc=0.90, cumulative_bound=0.05, absolute_floor=0.92,
        ) == pytest.approx(0.92)


# ── CHANGE 2: bounded cosine-scheduled stabilization + step-count plumbing ────


def _build_real_trainer():
    from mimarsinan.model_training.basic_trainer import BasicTrainer
    from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

    factory = MockDataProviderFactory()
    dl_factory = DataLoaderFactory(factory, num_workers=0)
    model = make_tiny_supermodel()
    loss = nn.CrossEntropyLoss()
    return BasicTrainer(model, "cpu", dl_factory, lambda m, x, y: loss(m(x), y))


class TestTrainStepsReturnsStepCount:
    def test_returns_step_count_when_requested(self):
        """``return_steps=True`` → ``(accuracy, steps_run)``; the count equals the
        optimize calls (no break before max_steps when target unreachable)."""
        trainer = _build_real_trainer()
        steps = [0]
        orig_optimize = trainer._optimize

        def counting(x, y, optimizer, scaler):
            steps[0] += 1
            return orig_optimize(x, y, optimizer, scaler)

        trainer._optimize = counting
        trainer.validate_n_batches = lambda n: 0.5  # never reaches target
        trainer.test = lambda: 0.5

        out = trainer.train_steps_until_target(
            lr=0.001, max_steps=7, target_accuracy=2.0,
            validation_n_batches=1, check_interval=100, patience=99, min_steps=99,
            return_steps=True,
        )
        assert isinstance(out, tuple) and len(out) == 2
        acc, n_steps = out
        assert acc == pytest.approx(0.5)
        assert n_steps == steps[0] == 7

    def test_default_returns_accuracy_only_byte_identical(self):
        """Default (no ``return_steps``) returns the accuracy float exactly as
        today — the pinned contract for existing callers."""
        trainer = _build_real_trainer()
        trainer.validate_n_batches = lambda n: 0.5
        trainer.test = lambda: 0.5
        out = trainer.train_steps_until_target(
            lr=0.001, max_steps=3, target_accuracy=2.0,
            validation_n_batches=1, check_interval=100, patience=99, min_steps=99,
        )
        assert out == pytest.approx(0.5)
        assert not isinstance(out, tuple)


class _StabCountTuner(SmoothAdaptationTuner):
    """Records every ``train_steps_until_target`` call and the cosine LR trace."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.train_calls = []
        self.lr_trace = []

        def _fake_train(lr, max_steps, target, *args, **kwargs):
            self.train_calls.append({"lr": lr, "max_steps": int(max_steps), "kwargs": kwargs})
            return None

        self.trainer.train_steps_until_target = _fake_train
        self.trainer.validate = lambda: 0.9
        self.trainer.validate_n_batches = lambda n: 0.9

    def _update_and_evaluate(self, rate):
        return 0.9

    def _find_lr(self):
        return 0.001


class TestBoundedCosineStabilization:
    def _tuner(self, tmp_path, monkeypatch=None, **policy_overrides):
        if policy_overrides:
            override_tuning_policy(monkeypatch, **policy_overrides)
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        t = _StabCountTuner(
            MockPipeline(config=cfg, working_directory=str(tmp_path)),
            make_tiny_supermodel(), 0.9, 0.001,
        )
        t._committed_rate = 1.0
        t._validation_baseline = 0.9
        t._pipeline_hard_floor = None
        t._cached_lr = 0.0005
        return t

    def test_bounded_pass_uses_ratio_of_gradual_steps(self, tmp_path, deterministic_rng, monkeypatch):
        t = self._tuner(
            tmp_path,
            monkeypatch,
            stabilization_bounded=True,
            stabilization_ratio=0.5,
        )
        t._gradual_train_steps = 400  # accumulated across the gradual cycles
        t._stabilize_at_full_rate()
        assert len(t.train_calls) == 1, "bounded stabilization is a SINGLE pass"
        assert t.train_calls[0]["max_steps"] == 200, "N = int(0.5 * 400)"
        tuner_kwargs = t.train_calls[0]["kwargs"]
        # Hard cutoff: no patience extension, no min-improvement plateau rounds.
        assert tuner_kwargs.get("patience", 0) <= 1
        t.close()

    def test_bounded_pass_lr_decays_cosine_to_near_zero(self, tmp_path, deterministic_rng, monkeypatch):
        """The bounded pass schedules a CosineAnnealingLR from the chosen lr down
        to ~0 over exactly N steps. Drive the real cosine schedule and assert the
        LR monotonically decays to near zero by the last step."""
        override_tuning_policy(
            monkeypatch, stabilization_bounded=True, stabilization_ratio=1.0
        )
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        trainer = _build_real_trainer()

        observed = []
        orig_optimize = trainer._optimize

        def capture(x, y, optimizer, scaler):
            observed.append(optimizer.param_groups[0]["lr"])
            return orig_optimize(x, y, optimizer, scaler)

        trainer._optimize = capture
        trainer.validate_n_batches = lambda n: 0.9
        trainer.validate = lambda: 0.9
        trainer.test = lambda: 0.9

        from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner

        class _RealStabTuner(SmoothAdaptationTuner):
            def _update_and_evaluate(self, rate):
                return 0.9

            def _find_lr(self):
                return 0.01

        t = _RealStabTuner(
            MockPipeline(config=cfg, working_directory=str(tmp_path)),
            make_tiny_supermodel(), 0.9, 0.001,
        )
        t.trainer = trainer
        t._committed_rate = 1.0
        t._validation_baseline = 0.9
        t._pipeline_hard_floor = None
        t._cached_lr = 0.01
        t._rollback_tolerance = 0.05
        t._gradual_train_steps = 12

        t._stabilize_at_full_rate()

        assert len(observed) == 12, f"hard cutoff at N=12 steps, ran {len(observed)}"
        assert observed[0] > observed[-1], "LR must decay over the pass"
        assert observed[-1] < observed[0] * 0.1, "cosine decays to near zero"
        t.close()

    def test_flag_off_reproduces_round_based_stabilization(self, tmp_path, deterministic_rng, monkeypatch):
        """Flag off → the existing patience/round-based stabilization (2x budget,
        cached LR), unchanged."""
        t = self._tuner(tmp_path, monkeypatch, stabilization_bounded=False)
        t._gradual_train_steps = 400
        t._stabilize_at_full_rate()
        assert len(t.train_calls) == 1
        assert t.train_calls[0]["max_steps"] == 2 * t._budget.max_training_steps
        assert t.train_calls[0]["lr"] == 0.0005
        t.close()


class TestGradualStepAccumulation:
    def test_run_accumulates_gradual_steps_when_bounded(self, tmp_path, deterministic_rng, monkeypatch):
        """A full scripted run accumulates the gradual training steps so the
        bounded stabilization has a budget to scale; flag off keeps it at 0."""
        override_tuning_policy(
            monkeypatch, stabilization_bounded=True, stabilization_ratio=0.5
        )
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))

        steps_per_cycle = [37]

        tuner = make_scripted_run_tuner(
            pipeline, make_tiny_supermodel(),
            instant_fn=lambda r: 0.87, post_fn=lambda r: 0.9,
        )

        # Report a fixed step count from each gradual recovery call.
        def _train(*a, **k):
            return (0.9, steps_per_cycle[0]) if k.get("return_steps") else 0.9

        tuner.trainer.train_steps_until_target = _train
        tuner.run()
        assert tuner._gradual_train_steps >= steps_per_cycle[0], (
            "gradual steps must accumulate across the ramp cycles"
        )
        tuner.close()


# ── CHANGE 3: tighter plateau detection via a check-interval divisor ──────────


class _PlateauTrainer:
    """Counts optimize calls; never improves → plateau-driven break."""

    def __init__(self):
        self.trainer = _build_real_trainer()
        self.steps = 0
        orig = self.trainer._optimize

        def counting(x, y, optimizer, scaler):
            self.steps += 1
            return orig(x, y, optimizer, scaler)

        self.trainer._optimize = counting
        self.trainer.validate_n_batches = lambda n: 0.5  # never improves
        self.trainer.test = lambda: 0.5


class TestTightPlateau:
    def test_divisor_detects_plateau_in_fewer_steps(self):
        """A larger divisor → smaller effective check interval → the stale-streak
        patience trips after FEWER optimize steps."""
        from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine

        def _run(check_interval):
            pt = _PlateauTrainer()
            RecoveryEngine.train_to_target(
                pt.trainer, 0.001, 2.0,
                max_steps=500, validation_n_batches=1,
                check_interval=check_interval, patience=2,
                min_steps=0, min_improvement=1e-3,
            )
            return pt.steps

        base_interval = 30
        steps_div1 = _run(base_interval)            # divisor 1 → interval 30
        steps_div3 = _run(max(1, base_interval // 3))  # divisor 3 → interval 10
        assert steps_div3 < steps_div1, (
            f"tighter interval must break sooner ({steps_div3} vs {steps_div1})"
        )

    def test_cycle_applies_divisor_to_check_interval_when_on(self, tmp_path, deterministic_rng, monkeypatch):
        """With the flag on and divisor>1, the recovery call receives a check
        interval reduced by the divisor; flag off passes the budget interval."""
        from unittest.mock import patch
        from mimarsinan.tuning.orchestration import smooth_adaptation_cycle as cyc

        captured = {}

        def fake_train_to_target(trainer, lr, target, **kw):
            captured.update(kw)

        tuner = _make_cycle_tuner(
            tmp_path, monkeypatch, tight_plateau=True, recovery_check_divisor=3
        )
        with patch.object(cyc.RecoveryEngine, "train_to_target", staticmethod(fake_train_to_target)):
            tuner.drive(0.3, validate_seq=[0.85, 0.90])
        base = tuner._budget.check_interval
        assert captured["check_interval"] == max(1, base // 3)
        tuner.close()

        captured.clear()
        tuner_off = _make_cycle_tuner(tmp_path, monkeypatch, tight_plateau=False)
        with patch.object(cyc.RecoveryEngine, "train_to_target", staticmethod(fake_train_to_target)):
            tuner_off.drive(0.3, validate_seq=[0.85, 0.90])
        assert captured["check_interval"] == tuner_off._budget.check_interval
        tuner_off.close()

    def test_divisor_combines_with_plateau_lr_reduction(self, tmp_path, deterministic_rng, monkeypatch):
        """The tight divisor and the ``recovery_lr_plateau`` reduction stack:
        the recovery call gets BOTH a reduced interval and the plateau kwargs."""
        from unittest.mock import patch
        from mimarsinan.tuning.orchestration import smooth_adaptation_cycle as cyc

        captured = {}

        def fake_train_to_target(trainer, lr, target, **kw):
            captured.update(kw)

        tuner = _make_cycle_tuner(
            tmp_path,
            monkeypatch,
            tight_plateau=True,
            recovery_check_divisor=3,
            recovery_lr_plateau=True,
            recovery_lr_plateau_factor=0.3,
            recovery_lr_plateau_reductions=2,
        )
        with patch.object(cyc.RecoveryEngine, "train_to_target", staticmethod(fake_train_to_target)):
            tuner.drive(0.3, validate_seq=[0.85, 0.90])
        assert captured["check_interval"] == max(1, tuner._budget.check_interval // 3)
        assert captured["plateau_lr_factor"] == pytest.approx(0.3)
        assert captured["plateau_lr_reductions"] == 2
        tuner.close()


# ── CHANGE 4: missed target is the sole LR-cache invalidation trigger ─────────


class TestRefindLrOnlyOnMiss:
    def test_miss_invalidates_lr_cache_when_flag_on(self, tmp_path, deterministic_rng, monkeypatch):
        tuner = _make_cycle_tuner(tmp_path, monkeypatch, refind_lr_on_miss=True)
        tuner._validation_baseline = 0.85  # abs floor 0.8075
        tuner.drive(0.3, validate_seq=[0.85, 0.81])  # commit but MISS (0.81 < 0.85)
        assert tuner._committed_rate == 0.3
        assert tuner.find_lr_calls == 1
        assert tuner._cached_lr is None, "miss with flag on invalidates the LR cache"
        tuner.drive(0.4, validate_seq=[0.85, 0.81])
        assert tuner.find_lr_calls == 2, "next cycle re-discovers the LR"
        tuner.close()

    def test_reached_target_never_invalidates(self, tmp_path, deterministic_rng, monkeypatch):
        tuner = _make_cycle_tuner(tmp_path, monkeypatch, refind_lr_on_miss=True)
        tuner.drive(0.3, validate_seq=[0.85, 0.90])  # post 0.90 >= 0.85 → reached
        assert tuner.find_lr_calls == 1
        assert tuner._cached_lr is not None, "reached target keeps the LR cache"
        tuner.close()

    def test_flag_off_keeps_cache_on_miss(self, tmp_path, deterministic_rng, monkeypatch):
        tuner = _make_cycle_tuner(tmp_path, monkeypatch, refind_lr_on_miss=False)
        tuner._validation_baseline = 0.85
        tuner.drive(0.3, validate_seq=[0.85, 0.81])
        assert tuner._committed_rate == 0.3
        assert tuner._cached_lr is not None, "flag off keeps the cached LR on a miss"
        tuner.drive(0.4, validate_seq=[0.85, 0.81])
        assert tuner.find_lr_calls == 1
        tuner.close()


# ── default-off regression: all new levers off → byte-identical decisions ─────


class TestAllLeversDefaultOff:
    def test_decisions_and_stabilization_byte_identical_when_off(self, tmp_path, deterministic_rng):
        # Default config (none of the new flags set) → original per-cycle behavior.
        tuner = _make_cycle_tuner(tmp_path)
        tuner._validation_baseline = 0.80  # floor 0.76, keeps abs floor out of the way

        # commit-then-miss: cache retained (CHANGE 4 off).
        tuner.drive(0.3, validate_seq=[0.85, 0.81])
        assert tuner._committed_rate == 0.3
        assert tuner.find_lr_calls == 1
        assert tuner._cached_lr is not None
        tuner.drive(0.4, validate_seq=[0.85, 0.81])
        assert tuner.find_lr_calls == 1

        # rollback gate relative-to-previous (CHANGE 1 off): small slips accumulate.
        assert tuner.drive(0.5, validate_seq=[0.90, 0.88]) == 0.5
        assert tuner.drive(0.6, validate_seq=[0.88, 0.86]) == 0.6
        assert tuner.drive(0.7, validate_seq=[0.86, 0.84]) == 0.7
        tuner.close()

    def test_recovery_call_uses_budget_interval_when_off(self, tmp_path, deterministic_rng):
        """CHANGE 3 off → the recovery call uses the budget check interval."""
        from unittest.mock import patch
        from mimarsinan.tuning.orchestration import smooth_adaptation_cycle as cyc

        captured = {}

        def fake_train_to_target(trainer, lr, target, **kw):
            captured.update(kw)

        tuner = _make_cycle_tuner(tmp_path)
        with patch.object(cyc.RecoveryEngine, "train_to_target", staticmethod(fake_train_to_target)):
            tuner.drive(0.3, validate_seq=[0.85, 0.90])
        assert captured["check_interval"] == tuner._budget.check_interval
        assert captured.get("plateau_lr_factor", 1.0) == pytest.approx(1.0)
        assert captured.get("plateau_lr_reductions", 0) == 0
        tuner.close()
