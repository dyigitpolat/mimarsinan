"""RecoveryEngine — the corrector: fine-tune to target with hook lifecycle."""

from __future__ import annotations


RESET_PER_CYCLE = "reset_per_cycle"
"""Build a fresh optimizer for every recovery call (Adam moments discarded)."""

PERSIST_WITHIN_CYCLE = "persist_within_cycle"
"""Reuse one caller-owned optimizer across recovery calls (moments persist)."""


class RecoveryEngine:
    """Train-to-target corrector with recovery-hook lifecycle management."""

    @staticmethod
    def train_to_target(trainer, lr, target, *, max_steps, hooks=None, optimizer=None, **kwargs):
        """``train_steps_until_target`` with the hooks removed in ``finally``.

        ``optimizer=None`` is the bit-exact fresh-build path; a supplied optimizer is
        reused (Adam moments persist). Extra ``kwargs`` forward to the trainer.
        """
        active_hooks = list(hooks) if hooks else []
        try:
            return trainer.train_steps_until_target(
                lr, max_steps, target, 0, optimizer=optimizer, **kwargs
            )
        finally:
            for hook in active_hooks:
                hook.remove()


class PersistentOptimizerOwner:
    """Lazily builds and owns one step optimizer for ``persist_within_cycle``.

    Rebuilds the optimizer whenever the model's parameter set changes identity, so
    moments persist for stable-param tuners but reset for tuners that replace
    parameters each cycle (a stale optimizer would step the wrong tensors).
    """

    def __init__(self, trainer):
        self._trainer = trainer
        self._optimizer = None
        self._param_ids = None

    def optimizer_for(self, lr):
        current = tuple(id(p) for p in self._trainer.model.parameters())
        if self._optimizer is None or current != self._param_ids:
            self._optimizer = self._trainer.build_step_optimizer(lr)
            self._param_ids = current
        return self._optimizer

    def reset(self):
        self._optimizer = None
        self._param_ids = None
