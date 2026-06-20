"""The uniform rate-tuner seam — the three driver-facing verbs every tuner exposes.

Fix A's real deliverable (``final_recommendations`` §A): not "move the fast ladder
up", but the *uniform rate-tuner seam* an ``OptimizationDriver`` consumes. Every
rate tuner — the KD-blend family (LIF / TTFS-cycle), the analytical
clamp/shift/activation-quant/weight-quant chain, the manager-rate family — exposes
the same three verbs, against which a driver (``controller | fast | …``) is generic:

* ``ramp(rate)``       — the predictor: apply transformation T at ``rate`` and read
  back the post-apply metric (the model state advances along the homotopy α-axis).
* ``recover_to(target)`` — the corrector: fine-tune toward an accuracy ``target``.
* ``probe()``          — read a validation metric without committing or training.

The verbs DELEGATE to each tuner's CURRENT methods (no behavior change): this is a
uniform façade over the legacy private methods, not a new control path. The tuner's
own ``run()`` loop is untouched — E2 unbinds the driver to drive ``run()`` through
this seam; E1 only defines and locks the seam (default-off, byte-identical).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RateTunerSeam(Protocol):
    """The driver-facing three-verb contract every rate tuner implements.

    An ``OptimizationDriver`` drives ANY tuner through these verbs uniformly; the
    tuner maps them onto its own rate-application / recovery / validation logic.
    """

    def ramp(self, rate: float):
        """Apply transformation T at ``rate`` and return the post-apply metric."""
        ...

    def recover_to(self, target: float):
        """Fine-tune the model toward accuracy ``target``; return the recovery result."""
        ...

    def probe(self) -> float:
        """Read a non-committing validation metric of the current model state."""
        ...

    def seam_descriptor(self) -> str:
        """A short identifier of the concrete seam mapping (for diagnostics)."""
        ...


class RateTunerSeamMixin:
    """Default seam for the ``SmoothAdaptationTuner`` family (the smooth tuners).

    The three verbs map onto the legacy private methods that already implement the
    predictor / corrector / probe so the seam is byte-identical to the cycle path:

    * ``ramp`` → ``_update_and_evaluate`` (apply T at rate + return progress metric).
    * ``recover_to`` → ``_recover_to_target`` (the shared recovery-engine assembly
      the per-cycle ``_recover`` also calls — one SSOT, one statistical basis).
    * ``probe`` → ``trainer.validate_n_batches`` over the budget's eval batches.

    The mixin is intentionally thin and stateless: it owns no new state and never
    mutates the run loop, so mixing it into a tuner cannot change any number.
    """

    def ramp(self, rate: float):
        """Predictor: apply T at ``rate`` via the tuner's ``_update_and_evaluate``."""
        return self._update_and_evaluate(float(rate))

    def recover_to(self, target: float, rate: float = None):
        """Corrector: recover toward ``target`` via the shared recovery primitive.

        ``rate`` selects which recovery hooks are installed (pruning masks etc.);
        it defaults to the currently committed rate so the seam mirrors a cycle's
        recovery at the live ramp position."""
        ramp_rate = self._committed_rate if rate is None else float(rate)
        _, result = self._recover_to_target(float(target), ramp_rate)
        return result

    def probe(self) -> float:
        """Read the non-committing validation metric of the current model state."""
        return float(self.trainer.validate_n_batches(self._budget.eval_n_batches))

    def seam_descriptor(self) -> str:
        return f"{type(self).__name__}(smooth)"


class OneShotRateTunerSeamMixin:
    """Seam for the one-shot family (``ActivationShiftTuner`` — extends ``TunerBase``,
    not ``SmoothAdaptationTuner``: it has no per-cycle ramp loop).

    Its shape resists the smooth mixin (no ``_update_and_evaluate`` / ``_budget``
    cycle eval / ``_recover_to_target``), so the seam is given over its EXISTING
    controller methods (SAFE INCREMENT): ``ramp`` applies the full shift through its
    axis, ``recover_to`` runs its step-budgeted ``train_steps_until_target`` recovery
    with the cached LR, and ``probe`` is the trainer validate. No behavior change —
    these delegate to exactly the calls ``run()`` already makes.
    """

    def ramp(self, rate: float):
        """Apply the one-shot transform at ``rate`` (1.0 = the full shift)."""
        self._axis.set_rate(float(rate))
        return None

    def recover_to(self, target: float, rate: float = None):
        """Step-budgeted recovery toward ``target`` — the tuner's ``run`` recovery."""
        lr = self._find_lr()
        return self.trainer.train_steps_until_target(
            lr,
            self._budget.max_training_steps,
            float(target),
            0,
            validation_n_batches=self._budget.progress_eval_batches,
            check_interval=self._budget.check_interval,
            patience=5,
            min_steps=self._budget.check_interval * 3,
            min_improvement=self._budget.accuracy_se(),
        )

    def probe(self) -> float:
        return float(self.trainer.validate())

    def seam_descriptor(self) -> str:
        return f"{type(self).__name__}(one_shot)"
