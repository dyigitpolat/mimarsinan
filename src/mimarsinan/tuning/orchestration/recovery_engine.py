"""RecoveryEngine — the corrector: fine-tune to target with hook lifecycle.

Centralizes the train-to-target + recovery-hook install/remove-in-finally
pattern that was duplicated across the per-cycle recovery, the below-floor
safety net, and the full-rate stabilization pass. ``train_to_target`` threads an
optional caller-owned optimizer: ``optimizer=None`` keeps the historical
fresh-build-then-del path (bit-exact); a supplied optimizer is reused and never
deleted, so its Adam moments persist across recovery calls (P6,
``persist_within_cycle``). The remaining behavior-changing extensions (LR
discovery, tunable-param folding) layer on later behind flags.
"""

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

        ``kwargs`` (validation_n_batches / check_interval / patience / min_steps /
        min_improvement) forward to the trainer. ``optimizer=None`` is the
        bit-exact fresh-build path; a supplied optimizer is reused (moments persist).
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
    """Lazily builds and owns one step optimizer for ``persist_within_cycle``."""

    def __init__(self, trainer):
        self._trainer = trainer
        self._optimizer = None

    def optimizer_for(self, lr):
        if self._optimizer is None:
            self._optimizer = self._trainer.build_step_optimizer(lr)
        return self._optimizer

    def reset(self):
        self._optimizer = None
