"""SmoothAdaptationTuner for a single AdaptationManager rate field."""

from __future__ import annotations

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class AdaptationRateTuner(SmoothAdaptationTuner):
    """Drive one ``adaptation_manager.<rate_attr>`` across all perceptrons."""

    rate_attr: str = "quantization_rate"

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager

    def _get_extra_state(self):
        return getattr(self.adaptation_manager, self.rate_attr)

    def _apply_rate(self, rate) -> None:
        setattr(self.adaptation_manager, self.rate_attr, rate)
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def _set_extra_state(self, extra):
        self._apply_rate(extra)

    def _update_and_evaluate(self, rate):
        self._apply_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _after_run(self):
        self._continue_to_full_rate()
        setattr(self.adaptation_manager, self.rate_attr, 1.0)
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        self._committed_rate = 1.0
