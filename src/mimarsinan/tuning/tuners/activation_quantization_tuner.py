"""Tuner for gradual activation quantization."""

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class ActivationQuantizationTuner(SmoothAdaptationTuner):
    def __init__(self, pipeline, model, target_tq, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.target_tq = target_tq
        self.adaptation_manager = adaptation_manager
        self._final_metric = None

    def _get_extra_state(self):
        return self.adaptation_manager.quantization_rate

    def _set_extra_state(self, extra):
        self.adaptation_manager.quantization_rate = extra
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)

    def _update_and_evaluate(self, rate):
        self.adaptation_manager.quantization_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()

    def _after_run(self):
        self._continue_to_full_rate()

        self.adaptation_manager.quantization_rate = 1.0
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)

        recovered_val = self._ensure_validation_threshold()
        if recovered_val >= self._validation_floor_for_commit():
            self._committed_rate = 1.0
        self._final_metric = recovered_val
        self._flush_enforcement_hooks()
        return self._final_metric

    def run(self):
        return super().run()
