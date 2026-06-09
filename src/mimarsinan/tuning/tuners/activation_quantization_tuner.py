"""Tuner for gradual activation quantization."""

from mimarsinan.tuning.adaptation_rate_tuner import AdaptationRateTuner


class ActivationQuantizationTuner(AdaptationRateTuner):
    rate_attr = "quantization_rate"
    _budget_multiplier = 2.0

    def __init__(self, pipeline, model, target_tq, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr, adaptation_manager)
        self.target_tq = target_tq
        self._final_metric = None

    def _stabilization_budget(self):
        return 4 * int(self._budget.max_training_steps)

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
