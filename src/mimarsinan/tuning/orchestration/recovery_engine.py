"""RecoveryEngine — the corrector: fine-tune to target with hook lifecycle.

Centralizes the train-to-target + recovery-hook install/remove-in-finally
pattern that was duplicated across the per-cycle recovery, the below-floor
safety net, and the full-rate stabilization pass. ``train_to_target`` is a pure,
bit-exact extraction (same ``train_steps_until_target`` args, same finally
teardown). LR discovery (cached on the tuner today), persistent optimizer state
(``optimizer_policy``), and folding ``axis.tunable_parameters()`` into the
optimized params are the behavior-changing extensions layered on later behind a
flag — keeping this surface bit-exact for the golden-trace gate.
"""

from __future__ import annotations


class RecoveryEngine:
    """Train-to-target corrector with recovery-hook lifecycle management."""

    @staticmethod
    def train_to_target(
        trainer,
        lr,
        target,
        *,
        max_steps,
        validation_n_batches,
        check_interval,
        patience,
        min_steps,
        min_improvement,
        hooks,
    ) -> None:
        """``train_steps_until_target`` with the hooks removed in ``finally``."""
        try:
            trainer.train_steps_until_target(
                lr,
                max_steps,
                target,
                0,
                validation_n_batches=validation_n_batches,
                check_interval=check_interval,
                patience=patience,
                min_steps=min_steps,
                min_improvement=min_improvement,
            )
        finally:
            for hook in hooks:
                hook.remove()
