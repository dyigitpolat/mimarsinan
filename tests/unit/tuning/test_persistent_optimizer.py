"""P6: persistent optimizer state across recovery calls.

Feature flag: ``tuning_persist_optimizer`` (default False).

Default (``reset_per_cycle``): every recovery call builds a fresh optimizer and
deletes it — Adam moments are discarded each call (bit-exact historical path).
Recovery passes ``optimizer=None`` to the trainer.

Persist (``persist_within_cycle``): the cycle owns ONE optimizer and threads the
same object through every recovery call, so Adam moments survive across calls.

The hierarchy under test:
- ``basic_trainer_steps`` step APIs accept an optional externally-owned
  optimizer (None => build fresh + del; provided => use + keep).
- ``RecoveryEngine.train_to_target`` wraps the trainer call and removes hooks in
  a finally (even on exception), threading the optimizer through.
- ``SmoothAdaptationCycleMixin._recovery_optimizer`` selects None vs an owned
  optimizer from ``_optimizer_policy()``, gated by the config flag.

All trainer interactions use stubs — no real training is required.
"""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    CONFIG_KEYS_SET,
)
from mimarsinan.model_training import basic_trainer_steps
from mimarsinan.tuning.orchestration.recovery_engine import (
    PERSIST_WITHIN_CYCLE,
    RESET_PER_CYCLE,
    PersistentOptimizerOwner,
    RecoveryEngine,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


# ---------------------------------------------------------------------------
# Stub trainer — records optimizer plumbing without running real training.
# ---------------------------------------------------------------------------

class _Sentinel:
    """Stand-in optimizer object; identity is what we assert on."""


class _FakeTensor:
    """Tensor-like batch element; ``.to(device)`` is a no-op returning self."""

    def to(self, device):
        return self


class _StubStepTrainer:
    """Minimal trainer exercising the ``basic_trainer_steps`` optimizer plumbing.

    Drives a fixed number of steps over a one-batch fake iterator. Records which
    optimizer-construction path ran and the optimizer object each step saw.
    """

    def __init__(self):
        self.device = "cpu"
        self.model = nn.Linear(2, 2)
        self.train_loader = [(_FakeTensor(), _FakeTensor())]
        self.train_iter = iter(self.train_loader)
        self.fresh_build_calls = 0
        self.scheduler_for_optimizer_calls = 0
        self.optimizers_seen = []
        self.report_calls = []

    # -- basic_trainer_steps hooks --

    def _get_optimizer_and_scheduler_steps(self, lr, total_steps, *, constant_lr=False):
        self.fresh_build_calls += 1
        opt = _Sentinel()
        opt.param_groups = [{"lr": lr}]
        return opt, _FakeScheduler(), _FakeScaler()

    def _scheduler_and_scaler_for_optimizer(self, optimizer, lr, total_steps, *, constant_lr=False):
        self.scheduler_for_optimizer_calls += 1
        return _FakeScheduler(), _FakeScaler()

    def _optimize(self, x, y, optimizer, scaler):
        self.optimizers_seen.append(optimizer)

    def _report(self, name, value):
        self.report_calls.append((name, value))

    def next_training_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

    def validate_n_batches(self, n):
        return 0.5


class _FakeScheduler:
    def step(self):
        pass


class _FakeScaler:
    pass


# ---------------------------------------------------------------------------
# basic_trainer_steps: None => fresh+del (bit-exact); provided => reuse+keep.
# ---------------------------------------------------------------------------

class TestStepAPIsOptimizerOwnership:
    def test_train_n_steps_default_builds_fresh(self):
        trainer = _StubStepTrainer()
        basic_trainer_steps.train_n_steps(trainer, lr=0.01, steps=3, optimizer=None)

        assert trainer.fresh_build_calls == 1
        assert trainer.scheduler_for_optimizer_calls == 0

    def test_train_n_steps_uses_provided_optimizer(self):
        trainer = _StubStepTrainer()
        owned = _Sentinel()
        owned.param_groups = [{"lr": 0.01}]

        basic_trainer_steps.train_n_steps(trainer, lr=0.01, steps=3, optimizer=owned)

        # No fresh build; scheduler built around the supplied optimizer.
        assert trainer.fresh_build_calls == 0
        assert trainer.scheduler_for_optimizer_calls == 1
        # Every step optimized through the *same* owned object.
        assert trainer.optimizers_seen == [owned, owned, owned]

    def test_train_steps_until_target_default_builds_fresh(self):
        trainer = _StubStepTrainer()
        out = basic_trainer_steps.train_steps_until_target(
            trainer,
            lr=0.01,
            max_steps=2,
            target_accuracy=1.0,
            optimizer=None,
        )
        assert trainer.fresh_build_calls == 1
        assert trainer.scheduler_for_optimizer_calls == 0
        assert out == 0.5  # final validate_n_batches

    def test_train_steps_until_target_reuses_provided_optimizer(self):
        trainer = _StubStepTrainer()
        owned = _Sentinel()
        owned.param_groups = [{"lr": 0.01}]

        basic_trainer_steps.train_steps_until_target(
            trainer,
            lr=0.01,
            max_steps=2,
            target_accuracy=1.0,
            optimizer=owned,
        )
        assert trainer.fresh_build_calls == 0
        assert trainer.scheduler_for_optimizer_calls == 1
        assert all(o is owned for o in trainer.optimizers_seen)


# ---------------------------------------------------------------------------
# RecoveryEngine: hooks removed in finally; optimizer threaded through.
# ---------------------------------------------------------------------------

class TestRecoveryEngine:
    def test_threads_optimizer_and_removes_hooks(self):
        recorded = {}
        hook = MagicMock()

        class _T:
            def train_steps_until_target(self, lr, max_steps, target, warmup, *, optimizer=None, **kw):
                recorded["lr"] = lr
                recorded["optimizer"] = optimizer
                recorded["kw"] = kw
                return "done"

        out = RecoveryEngine.train_to_target(
            _T(),
            lr=0.001,
            max_steps=5,
            target=0.9,
            hooks=[hook],
            optimizer="OWNED",
            patience=4,
        )
        assert out == "done"
        assert recorded["optimizer"] == "OWNED"
        assert recorded["lr"] == 0.001
        assert recorded["kw"] == {"patience": 4}
        hook.remove.assert_called_once()

    def test_removes_hooks_even_on_exception(self):
        hook = MagicMock()

        class _T:
            def train_steps_until_target(self, *a, **kw):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            RecoveryEngine.train_to_target(
                _T(), lr=0.001, max_steps=5, target=0.9, hooks=[hook]
            )
        hook.remove.assert_called_once()

    def test_none_hooks_is_safe(self):
        class _T:
            def train_steps_until_target(self, *a, **kw):
                return None

        RecoveryEngine.train_to_target(_T(), lr=0.001, max_steps=1, target=0.5, hooks=None)


class TestPersistentOptimizerOwner:
    def test_builds_once_and_reuses(self):
        builds = []

        class _T:
            def build_step_optimizer(self, lr):
                opt = _Sentinel()
                builds.append(lr)
                return opt

        owner = PersistentOptimizerOwner(_T())
        a = owner.optimizer_for(0.01)
        b = owner.optimizer_for(0.005)  # different lr, same object
        assert a is b
        assert builds == [0.01]  # built exactly once

    def test_reset_drops_owned_optimizer(self):
        class _T:
            def build_step_optimizer(self, lr):
                return _Sentinel()

        owner = PersistentOptimizerOwner(_T())
        a = owner.optimizer_for(0.01)
        owner.reset()
        b = owner.optimizer_for(0.01)
        assert a is not b


# ---------------------------------------------------------------------------
# Cycle policy: flag selects reset_per_cycle (None) vs persist (shared object).
# ---------------------------------------------------------------------------

class _PolicyTuner(SmoothAdaptationTuner):
    """Concrete tuner that records the optimizer threaded into recovery."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.recovery_optimizers = []
        self.build_step_optimizer_calls = 0

    def _update_and_evaluate(self, rate):
        return 0.9

    def _find_lr(self):
        return 0.001

    def _wire(self):
        tuner = self

        def _fake_train(lr, max_steps, target, warmup, *, optimizer=None, **kwargs):
            tuner.recovery_optimizers.append(optimizer)
            return None

        self.trainer.train_steps_until_target = _fake_train
        self.trainer.validate = lambda: 0.9
        self.trainer.validate_n_batches = lambda n: 0.9

        orig_build = self.trainer.build_step_optimizer

        def _counting_build(lr):
            tuner.build_step_optimizer_calls += 1
            return orig_build(lr)

        self.trainer.build_step_optimizer = _counting_build


def _make_tuner(tmp_path, *, persist):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["tuning_persist_optimizer"] = persist
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    tuner = _PolicyTuner(pipeline, make_tiny_supermodel(), target_accuracy=0.9, lr=0.001)
    tuner._wire()
    tuner._rollback_tolerance = 0.05
    tuner._pipeline_hard_floor = None
    tuner._committed_rate = 0.0
    return tuner


class TestCycleOptimizerPolicy:
    def test_default_policy_is_reset_per_cycle(self, tmp_path):
        tuner = _make_tuner(tmp_path, persist=False)
        assert tuner._optimizer_policy() == RESET_PER_CYCLE

    def test_flag_selects_persist_policy(self, tmp_path):
        tuner = _make_tuner(tmp_path, persist=True)
        assert tuner._optimizer_policy() == PERSIST_WITHIN_CYCLE

    def test_default_path_passes_optimizer_none(self, tmp_path):
        """Flag-off recovery threads optimizer=None (bit-exact fresh-build path)."""
        tuner = _make_tuner(tmp_path, persist=False)

        tuner._adaptation(0.5)
        tuner._adaptation(0.6)

        assert tuner.recovery_optimizers == [None, None]
        assert tuner.build_step_optimizer_calls == 0

    def test_persist_path_reuses_same_optimizer(self, tmp_path):
        """Flag-on: both recover calls thread the SAME owned optimizer object,
        so Adam moments are not discarded between calls."""
        tuner = _make_tuner(tmp_path, persist=True)

        tuner._adaptation(0.5)
        tuner._adaptation(0.6)

        assert len(tuner.recovery_optimizers) == 2
        first, second = tuner.recovery_optimizers
        assert first is not None
        assert first is second, "persist policy must reuse one optimizer object"
        assert tuner.build_step_optimizer_calls == 1, "optimizer built exactly once"

    def test_safety_net_recovery_also_persists(self, tmp_path):
        """The below-floor safety net shares the same owned optimizer."""
        import warnings

        tuner = _make_tuner(tmp_path, persist=True)
        tuner._pipeline_hard_floor = 0.95  # force recovery attempts

        # one adaptation builds + owns the optimizer
        tuner._adaptation(0.5)
        built_after_adaptation = list(tuner.recovery_optimizers)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tuner._attempt_recovery_if_below_floor()

        net_optimizers = tuner.recovery_optimizers[len(built_after_adaptation):]
        assert net_optimizers, "safety net should have run recovery"
        assert all(o is built_after_adaptation[0] for o in net_optimizers)
        assert tuner.build_step_optimizer_calls == 1

    def test_new_construct_via_new_does_not_crash_policy(self, tmp_path):
        """Tuners built via __new__ (bypassing __init__) must still answer
        _recovery_optimizer through the getattr-guarded owner slot."""
        tuner = _make_tuner(tmp_path, persist=True)
        # Simulate the bypassed-attr path: owner slot absent.
        if hasattr(tuner, "_persistent_optimizer_owner"):
            del tuner._persistent_optimizer_owner
        opt = tuner._recovery_optimizer(0.001)
        assert opt is not None
        assert tuner._recovery_optimizer(0.001) is opt


# ---------------------------------------------------------------------------
# Config flag registration.
# ---------------------------------------------------------------------------

class TestConfigFlagRegistered:
    def test_flag_default_off(self):
        assert DEFAULT_DEPLOYMENT_PARAMETERS["tuning_persist_optimizer"] is False

    def test_flag_in_config_keys_set(self):
        assert "tuning_persist_optimizer" in CONFIG_KEYS_SET
