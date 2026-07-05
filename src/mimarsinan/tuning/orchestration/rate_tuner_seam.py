"""The uniform rate-tuner seam — the three driver-facing verbs every tuner exposes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase
from mimarsinan.tuning.orchestration.tuning_budget import TuningBudget


@runtime_checkable
class RateTunerSeam(Protocol):
    """The driver-facing three-verb contract every rate tuner implements."""

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
    """Default seam for the ``SmoothAdaptationTuner`` family.

    The three verbs map onto the legacy private methods (``_update_and_evaluate`` /
    ``_recover_to_target`` / ``validate_n_batches``); stateless, so mixing it into a
    tuner cannot change any number.
    """

    if TYPE_CHECKING:
        # Host contract: supplied by SmoothAdaptationCycleMixin + TunerBase.
        # ``Any`` because tuner families narrow the trainer type (invariance).
        trainer: Any
        _budget: TuningBudget
        _committed_rate: float

        def _update_and_evaluate(self, rate) -> float: ...
        def _recover_to_target(self, target, rate) -> tuple[float | None, Any]: ...

    def ramp(self, rate: float):
        """Predictor: apply T at ``rate`` via the tuner's ``_update_and_evaluate``."""
        return self._update_and_evaluate(float(rate))

    def recover_to(self, target: float, rate: float | None = None):
        """Corrector: recover toward ``target`` via the shared recovery primitive.

        ``rate`` (default: the committed rate) selects which recovery hooks install;
        the discovered LR is stashed on ``self._last_recover_lr`` and the return is
        the recovery result (the driver-facing contract).
        """
        ramp_rate = self._committed_rate if rate is None else float(rate)
        lr, result = self._recover_to_target(float(target), ramp_rate)
        self._last_recover_lr = lr
        return result

    def probe(self) -> float:
        """Read the non-committing validation metric of the current model state."""
        return float(self.trainer.validate_n_batches(self._budget.eval_n_batches))

    def seam_descriptor(self) -> str:
        return f"{type(self).__name__}(smooth)"


class OneShotRateTunerSeamMixin:
    """Seam for the one-shot family (``ActivationShiftTuner``, which has no per-cycle
    ramp loop). The verbs delegate to its existing controller methods: ``ramp``
    applies the full shift, ``recover_to`` runs ``train_steps_until_target``.
    """

    if TYPE_CHECKING:
        # Host contract: supplied by the owning one-shot tuner + TunerBase.
        trainer: BasicTrainer
        _budget: TuningBudget
        _axis: AdaptationAxisBase

        def _find_lr(self) -> float: ...

    def ramp(self, rate: float):
        """Apply the one-shot transform at ``rate`` (1.0 = the full shift)."""
        self._axis.set_rate(float(rate))
        return None

    def _train_recovery(self, target: float, *, final_validation: bool = True):
        """The one-shot recovery invocation, with the fix-C refusal fallback:
        a refused LR search skips training and returns the entry probe."""
        lr = self._find_lr()
        if lr is None:
            return self.trainer.validate_n_batches(
                self._budget.progress_eval_batches
            )
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
            final_validation=final_validation,
        )

    def recover_to(self, target: float, rate: float | None = None):
        """Step-budgeted recovery toward ``target`` — the tuner's ``run`` recovery."""
        return self._train_recovery(float(target))

    def probe(self) -> float:
        return float(self.trainer.validate())

    def seam_descriptor(self) -> str:
        return f"{type(self).__name__}(one_shot)"
