"""RecoveryEngine.train_to_target: arg passthrough + hook teardown (P3b)."""

import pytest

from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine


class _Hook:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _Trainer:
    def __init__(self, raise_exc=False):
        self.calls = []
        self._raise = raise_exc

    def train_steps_until_target(self, lr, max_steps, target, start, *, optimizer=None, **kw):
        self.calls.append((lr, max_steps, target, start, optimizer, kw))
        if self._raise:
            raise RuntimeError("boom")


def test_passes_exact_args_and_removes_hooks():
    trainer = _Trainer()
    hooks = [_Hook(), _Hook()]
    RecoveryEngine.train_to_target(
        trainer, 1e-3, 0.9,
        max_steps=100, validation_n_batches=16, check_interval=10,
        patience=5, min_steps=30, min_improvement=0.005, hooks=hooks,
    )
    (lr, max_steps, target, start, optimizer, kw) = trainer.calls[0]
    assert (lr, max_steps, target, start) == (1e-3, 100, 0.9, 0)
    assert optimizer is None  # default reset_per_cycle path
    assert kw == {
        "validation_n_batches": 16, "check_interval": 10,
        "patience": 5, "min_steps": 30, "min_improvement": 0.005,
    }
    assert all(h.removed for h in hooks)


def test_threads_owned_optimizer():
    trainer = _Trainer()
    RecoveryEngine.train_to_target(
        trainer, 1e-3, 0.9, max_steps=10, hooks=None, optimizer="OWNED", patience=4,
    )
    (_, _, _, _, optimizer, kw) = trainer.calls[0]
    assert optimizer == "OWNED"
    assert kw == {"patience": 4}


def test_hooks_removed_even_on_exception():
    trainer = _Trainer(raise_exc=True)
    hooks = [_Hook(), _Hook()]
    with pytest.raises(RuntimeError):
        RecoveryEngine.train_to_target(
            trainer, 1e-3, 0.9,
            max_steps=100, validation_n_batches=16, check_interval=10,
            patience=5, min_steps=30, min_improvement=0.005, hooks=hooks,
        )
    assert all(h.removed for h in hooks)
