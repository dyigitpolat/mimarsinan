"""SmoothAdaptationTuner for a single AdaptationManager rate field."""

from __future__ import annotations

from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner
from mimarsinan.tuning.perceptron_rate import apply_manager_rate


class AdaptationRateTuner(SmoothAdaptationTuner):
    """Drive one ``adaptation_manager.<rate_attr>`` across all perceptrons."""

    rate_attr: str = "quantization_rate"

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager

    def _get_extra_state(self):
        return getattr(self.adaptation_manager, self.rate_attr)

    def _apply_rate(self, rate) -> None:
        apply_manager_rate(
            self.model, self.adaptation_manager, self.pipeline.config,
            self.rate_attr, rate,
        )

    def _set_extra_state(self, extra):
        self._apply_rate(extra)

    def _update_and_evaluate(self, rate):
        self._apply_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _after_run(self):
        self._continue_to_full_rate()
        self._apply_rate(1.0)
        self._committed_rate = 1.0
        # Enforce the pipeline floor for the whole rate-tuner family (the
        # subclasses used to each re-add this identical safety net).
        self._final_metric = self._ensure_pipeline_threshold()
        return self._final_metric
