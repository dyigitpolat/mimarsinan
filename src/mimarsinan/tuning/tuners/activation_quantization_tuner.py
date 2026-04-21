"""Tuner for gradual activation quantization."""

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class ActivationQuantizationTuner(SmoothAdaptationTuner):
    # Aggressive activation quantization (small ``target_tq`` such as 4) is
    # a discrete, heavy transformation: intermediate rates create mixed
    # quantized/continuous activations that the model handles WORSE than
    # either endpoint, so the gradual ramp burns training steps without
    # lifting the rate=1.0 endpoint accuracy. Doubling the per-cycle
    # recovery budget gives the final rate=1.0 cycle enough runway to
    # cross the strict gate on a single attempt rather than scraping
    # across on the 8th cycle with barely any margin.
    _budget_multiplier = 2.0

    def __init__(self, pipeline, model, target_tq, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.target_tq = target_tq
        self.adaptation_manager = adaptation_manager
        self._final_metric = None

    def _stabilization_budget(self):
        # At rate=1.0 the model is fully quantized and needs extended
        # consolidation; the default 2x is dominated by the LR schedule's
        # tail. Give stabilization a full 4x so the best-state tracker can
        # actually find a stable peak instead of plateauing in noise.
        return 4 * int(self._budget.max_training_steps)

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

        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric

    def run(self):
        return super().run()
