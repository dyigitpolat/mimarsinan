"""Tests for the three coupled recovery-quality flags (all default-off byte-identical):

- CHANGE 1 ``tuning_refind_lr_on_miss``: a cycle that misses the target invalidates
  the cached LR so the next cycle re-discovers it.
- CHANGE 2 ``tuning_recovery_lr_plateau``: ``train_steps_until_target`` reduces the
  optimizer LR on a plateau (instead of breaking) until the reductions are exhausted.
- CHANGE 3 ``tuning_rollback_ratchet``: the per-cycle rollback gate is computed
  against the best-committed accuracy (a ratcheting high-water mark) so repeated
  small slips cannot accumulate down to the floor.

The default-off regression at the end asserts the three flags off reproduces the
current per-cycle decisions exactly.
"""

import pytest
import torch
import torch.nn as nn

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    default_config,
    MockDataProviderFactory,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner
from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor


# ── shared scripted tuner ────────────────────────────────────────────────────


class _ScriptedCycleTuner(SmoothAdaptationTuner):
    """Drives ``_adaptation`` with a scripted ``[pre, post]`` validate sequence.

    ``train_steps_until_target`` is stubbed out and ``_find_lr`` counts its calls,
    so the LR-cache behavior (CHANGE 1) and the rollback gate (CHANGE 3) can be
    exercised deterministically.
    """

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


def _make_cycle_tuner(tmp_path, **flags):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    cfg.update(flags)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    tuner = _ScriptedCycleTuner(pipeline, make_tiny_supermodel(), 0.9, 0.001)
    tuner._rollback_tolerance = 0.05
    tuner._validation_baseline = 0.9
    return tuner


# ── CHANGE 1: re-find LR on missed target ────────────────────────────────────


class TestRefindLrOnMiss:
    def test_miss_invalidates_lr_cache_when_flag_on(self, tmp_path, deterministic_rng):
        tuner = _make_cycle_tuner(tmp_path, tuning_refind_lr_on_miss=True)
        # Baseline 0.85 → absolute floor 0.85*0.95 = 0.8075. Post 0.81 commits
        # (>= max(0.85-0.05, 0.8075)=0.8075) but 0.81 < target 0.9-0.05 = 0.85 → MISS.
        tuner._validation_baseline = 0.85
        tuner.drive(0.3, validate_seq=[0.85, 0.81])
        assert tuner.find_lr_calls == 1
        assert tuner._committed_rate == 0.3, "must commit (not roll back) to reach the miss path"
        assert tuner._cached_lr is None, "miss with flag on must invalidate the LR cache"

        # Next cycle re-discovers the LR (a second _find_lr call).
        tuner.drive(0.4, validate_seq=[0.85, 0.81])
        assert tuner.find_lr_calls == 2
        tuner.close()

    def test_miss_retains_lr_cache_when_flag_off(self, tmp_path, deterministic_rng):
        tuner = _make_cycle_tuner(tmp_path, tuning_refind_lr_on_miss=False)
        tuner._validation_baseline = 0.85
        tuner.drive(0.3, validate_seq=[0.85, 0.81])
        assert tuner._committed_rate == 0.3
        assert tuner.find_lr_calls == 1
        assert tuner._cached_lr is not None, "miss with flag off keeps the cached LR"

        tuner.drive(0.4, validate_seq=[0.85, 0.81])
        assert tuner.find_lr_calls == 1, "cache retained → no second LR discovery"
        tuner.close()

    def test_reached_target_never_invalidates_via_miss_path(self, tmp_path, deterministic_rng):
        """Hitting the target keeps the cache regardless of the flag."""
        tuner = _make_cycle_tuner(tmp_path, tuning_refind_lr_on_miss=True)
        tuner.drive(0.3, validate_seq=[0.85, 0.90])  # post 0.90 >= 0.85 → reached
        assert tuner.find_lr_calls == 1
        assert tuner._cached_lr is not None
        tuner.close()


# ── CHANGE 2: plateau LR reduction within recovery ───────────────────────────


def _build_real_trainer():
    from mimarsinan.model_training.basic_trainer import BasicTrainer
    from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

    factory = MockDataProviderFactory()
    dl_factory = DataLoaderFactory(factory, num_workers=0)
    model = make_tiny_supermodel()
    loss = nn.CrossEntropyLoss()
    return BasicTrainer(model, "cpu", dl_factory, lambda m, x, y: loss(m(x), y))


class _PlateauSurface:
    """Validation returns ``low`` until the optimizer BASE LR (``initial_lr``,
    immune to the warmup transient) drops below ``thresh``, then ``high`` — i.e.
    the model only improves once the recovery has reduced the LR."""

    def __init__(self, optimizer_ref, *, low, high, thresh):
        self._opt = optimizer_ref
        self._low = low
        self._high = high
        self._thresh = thresh
        self.observed_lrs = []

    def _base_lr(self):
        if self._opt[0] is None:
            return 1.0
        g = self._opt[0].param_groups[0]
        return g.get("initial_lr", g["lr"])

    def __call__(self, _n):
        base = self._base_lr()
        self.observed_lrs.append(base)
        return self._high if base < self._thresh else self._low


class TestPlateauLrReduction:
    def test_reduction_continues_and_reaches_higher_best(self):
        """With factor<1/reductions>0, a plateau-then-improve-at-lower-LR surface
        reduces the LR and continues, reaching the higher accuracy."""
        trainer = _build_real_trainer()
        opt_ref = [None]
        orig_optimize = trainer._optimize

        def capture_optimize(x, y, optimizer, scaler):
            opt_ref[0] = optimizer
            return orig_optimize(x, y, optimizer, scaler)

        trainer._optimize = capture_optimize

        # base lr 0.1; one 0.3 reduction → 0.03 (>= 0.05? no, 0.03 < 0.05) so the
        # very first reduction crosses the threshold and unlocks ``high``.
        surface = _PlateauSurface(opt_ref, low=0.5, high=0.95, thresh=0.05)
        trainer.validate_n_batches = surface
        trainer.test = lambda: 0.5

        result = trainer.train_steps_until_target(
            lr=0.1,
            max_steps=200,
            target_accuracy=2.0,  # unreachable → always plateau-driven
            validation_n_batches=1,
            check_interval=1,
            patience=2,
            min_steps=0,
            min_improvement=1e-3,
            plateau_lr_factor=0.3,
            plateau_lr_reductions=2,
        )

        assert any(lr < 0.05 for lr in surface.observed_lrs), (
            "the recovery must have reduced the LR below the threshold"
        )
        assert result == pytest.approx(0.95), (
            f"after reducing the LR the surface improves to 0.95, got {result}"
        )

    def test_defaults_reproduce_break_on_patience(self):
        """factor=1.0/reductions=0 (defaults) break on patience exactly as today —
        no LR reduction, and the same step count as the legacy break."""
        steps_default = self._run_and_count(plateau_lr_factor=1.0, plateau_lr_reductions=0)
        steps_legacy = self._run_and_count()  # no plateau kwargs at all
        assert steps_default == steps_legacy, (
            f"defaults must equal the legacy break ({steps_default} vs {steps_legacy})"
        )

    def _run_and_count(self, **plateau_kwargs):
        trainer = _build_real_trainer()
        steps = [0]
        orig_optimize = trainer._optimize

        def counting(x, y, optimizer, scaler):
            steps[0] += 1
            return orig_optimize(x, y, optimizer, scaler)

        trainer._optimize = counting
        trainer.validate_n_batches = lambda n: 0.5  # never improves, never reaches
        trainer.test = lambda: 0.5

        trainer.train_steps_until_target(
            lr=0.001,
            max_steps=100,
            target_accuracy=1.0,
            validation_n_batches=1,
            check_interval=1,
            patience=2,
            min_steps=0,
            min_improvement=1e-3,
            **plateau_kwargs,
        )
        return steps[0]

    def test_does_not_reduce_below_zero_reductions(self):
        """With reductions exhausted, training breaks instead of reducing forever."""
        trainer = _build_real_trainer()
        opt_ref = [None]
        orig_optimize = trainer._optimize

        def capture_optimize(x, y, optimizer, scaler):
            opt_ref[0] = optimizer
            return orig_optimize(x, y, optimizer, scaler)

        trainer._optimize = capture_optimize

        # Track the BASE LR (``initial_lr``, immune to the warmup transient) at
        # each check; never improves → reductions exhaust, then it breaks.
        observed = []

        def tracking_validate(_n):
            if opt_ref[0] is not None:
                g = opt_ref[0].param_groups[0]
                observed.append(g.get("initial_lr", g["lr"]))
            return 0.5

        trainer.validate_n_batches = tracking_validate
        trainer.test = lambda: 0.5

        trainer.train_steps_until_target(
            lr=0.1,
            max_steps=500,
            target_accuracy=2.0,
            validation_n_batches=1,
            check_interval=1,
            patience=1,
            min_steps=0,
            min_improvement=1e-3,
            plateau_lr_factor=0.5,
            plateau_lr_reductions=2,
        )

        # base LR bottoms out at 0.1 * 0.5 * 0.5 = 0.025 (2 reductions), then the
        # run breaks — it must terminate (no infinite reduction loop).
        assert min(observed) == pytest.approx(0.025), (
            f"base LR must bottom out at the 2-reduction floor 0.025, "
            f"got {min(observed):.4g}"
        )


# ── CHANGE 3: non-accumulating rollback gate that ratchets ───────────────────


class TestRollbackRatchet:
    def test_drift_gated_against_high_water_mark(self, tmp_path, deterministic_rng):
        """A slowly drifting surface is gated against the BEST committed accuracy,
        so it cannot accumulate small slips down toward the floor."""
        # Baseline 0.80 → absolute floor 0.80*0.95 = 0.76, well below the gate, so
        # the ratchet (not the cumulative-drift floor) is what catches the drift.
        tuner = _make_cycle_tuner(tmp_path, tuning_rollback_ratchet=True)
        tuner._rollback_tolerance = 0.05
        tuner._validation_baseline = 0.80
        tuner._best_committed_acc = 0.90  # high-water mark from earlier cycles

        # Cycle 1: post 0.87. Against best 0.90 - 0.05 = 0.85 → 0.87 >= 0.85 commit.
        r1 = tuner.drive(0.3, validate_seq=[0.88, 0.87])
        assert r1 == 0.3
        assert tuner._best_committed_acc == pytest.approx(0.90), (
            "best must not drop below the prior high-water mark"
        )

        # Cycle 2: post 0.86. Relative-to-previous (0.87 - 0.05 = 0.82) would PASS,
        # but ratchet uses best 0.90 - 0.05 = 0.85 → 0.86 >= 0.85 still commits.
        r2 = tuner.drive(0.4, validate_seq=[0.87, 0.86])
        assert r2 == 0.4

        # Cycle 3: post 0.84. Relative-to-previous (0.86 - 0.05 = 0.81) would PASS
        # and the abs floor (0.76) would PASS, but best 0.90 - 0.05 = 0.85 →
        # 0.84 < 0.85 → ROLLBACK (the slip is caught before it accumulates).
        r3 = tuner.drive(0.5, validate_seq=[0.86, 0.84])
        assert r3 == 0.4, "ratchet must roll back the slip below best-margin"
        tuner.close()

    def test_new_high_ratchets_threshold_up(self, tmp_path, deterministic_rng):
        """A new committed high tightens the gate for subsequent cycles."""
        tuner = _make_cycle_tuner(tmp_path, tuning_rollback_ratchet=True)
        tuner._rollback_tolerance = 0.05
        tuner._validation_baseline = 0.80  # floor 0.76, below the gate
        tuner._best_committed_acc = 0.85

        # Cycle 1: post 0.95 commits and ratchets the high-water mark up to 0.95.
        tuner.drive(0.3, validate_seq=[0.85, 0.95])
        assert tuner._best_committed_acc == pytest.approx(0.95)

        # Cycle 2: post 0.88. Against the OLD high 0.85 - 0.05 = 0.80 it would
        # pass, but the ratcheted best 0.95 - 0.05 = 0.90 → 0.88 < 0.90 → ROLLBACK.
        r2 = tuner.drive(0.4, validate_seq=[0.95, 0.88])
        assert r2 == 0.3, "ratcheted-up threshold rolls back the regression"
        tuner.close()

    def test_flag_off_uses_relative_to_previous_gate(self, tmp_path, deterministic_rng):
        """With the flag off, the gate is relative to pre_cycle_acc exactly as today:
        a drifting surface commits each small slip (and would accumulate)."""
        tuner = _make_cycle_tuner(tmp_path, tuning_rollback_ratchet=False)
        tuner._rollback_tolerance = 0.05
        tuner._validation_baseline = 0.80  # floor 0.76, below the gate

        # Each cycle slips 0.02 < 0.05 margin off the PREVIOUS → all commit.
        assert tuner.drive(0.3, validate_seq=[0.90, 0.88]) == 0.3
        assert tuner.drive(0.4, validate_seq=[0.88, 0.86]) == 0.4
        assert tuner.drive(0.5, validate_seq=[0.86, 0.84]) == 0.5
        # 0.84 is below the best-margin 0.85 but the relative gate accepted it.
        tuner.close()

    def test_acceptance_sensor_ratchet_threshold_helper(self):
        """The non-stalling ratchet threshold is the MAX of three lower bounds:
        relative (pre - margin), best-bound (best - cumulative_bound), abs floor."""
        # relative 0.84, best-bound 0.85, no floor → best-bound 0.85 dominates.
        assert AcceptanceSensor.ratchet_threshold(
            pre_cycle_acc=0.89, rollback_tolerance=0.05,
            best_committed_acc=0.90, cumulative_bound=0.05, absolute_floor=None,
        ) == pytest.approx(0.85)
        # abs floor 0.87 dominates the best-bound 0.85 and relative 0.84.
        assert AcceptanceSensor.ratchet_threshold(
            pre_cycle_acc=0.89, rollback_tolerance=0.05,
            best_committed_acc=0.90, cumulative_bound=0.05, absolute_floor=0.87,
        ) == pytest.approx(0.87)


# ── default-off regression: all three flags off → unchanged per-cycle decisions


class TestAllFlagsDefaultOff:
    def test_flags_default_off_unchanged_decisions(self, tmp_path, deterministic_rng):
        # Default config (none of the flags set) → original behavior.
        # Baseline 0.80 (floor 0.76) keeps the cumulative-drift floor out of the way.
        tuner = _make_cycle_tuner(tmp_path)
        tuner._validation_baseline = 0.80
        # commit-then-miss: cache retained (CHANGE 1 off).
        tuner.drive(0.3, validate_seq=[0.85, 0.81])
        assert tuner._committed_rate == 0.3
        assert tuner.find_lr_calls == 1
        assert tuner._cached_lr is not None
        tuner.drive(0.4, validate_seq=[0.85, 0.81])
        assert tuner.find_lr_calls == 1

        # rollback gate relative-to-previous (CHANGE 3 off): small slips accumulate
        # past the best-margin (they would be caught by the ratchet, but it is off).
        assert tuner.drive(0.5, validate_seq=[0.90, 0.88]) == 0.5
        assert tuner.drive(0.6, validate_seq=[0.88, 0.86]) == 0.6
        assert tuner.drive(0.7, validate_seq=[0.86, 0.84]) == 0.7
        tuner.close()

    def test_recovery_caller_passes_default_plateau_kwargs_when_off(self, tmp_path, deterministic_rng):
        """With ``tuning_recovery_lr_plateau`` off, the recovery call passes the
        no-op plateau defaults (factor 1.0 / reductions 0)."""
        from unittest.mock import patch
        from mimarsinan.tuning.orchestration import smooth_adaptation_cycle as cyc

        tuner = _make_cycle_tuner(tmp_path, tuning_recovery_lr_plateau=False)
        tuner.trainer.train_steps_until_target = lambda *a, **k: None

        captured = {}

        def fake_train_to_target(trainer, lr, target, **kw):
            captured.update(kw)

        with patch.object(cyc.RecoveryEngine, "train_to_target", staticmethod(fake_train_to_target)):
            tuner.drive(0.3, validate_seq=[0.85, 0.90])

        assert captured.get("plateau_lr_factor", 1.0) == pytest.approx(1.0)
        assert captured.get("plateau_lr_reductions", 0) == 0
        tuner.close()

    def test_recovery_caller_passes_configured_plateau_kwargs_when_on(self, tmp_path, deterministic_rng):
        from unittest.mock import patch
        from mimarsinan.tuning.orchestration import smooth_adaptation_cycle as cyc

        tuner = _make_cycle_tuner(
            tmp_path,
            tuning_recovery_lr_plateau=True,
            tuning_recovery_lr_plateau_factor=0.3,
            tuning_recovery_lr_plateau_reductions=2,
        )
        tuner.trainer.train_steps_until_target = lambda *a, **k: None

        captured = {}

        def fake_train_to_target(trainer, lr, target, **kw):
            captured.update(kw)

        with patch.object(cyc.RecoveryEngine, "train_to_target", staticmethod(fake_train_to_target)):
            tuner.drive(0.3, validate_seq=[0.85, 0.90])

        assert captured["plateau_lr_factor"] == pytest.approx(0.3)
        assert captured["plateau_lr_reductions"] == 2
        tuner.close()
