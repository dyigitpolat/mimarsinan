"""Tuner for gradual activation quantization."""

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class ActivationQuantizationTuner(SmoothAdaptationTuner):
    def __init__(self, pipeline, model, target_tq, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.target_tq = target_tq
        self.adaptation_manager = adaptation_manager

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
        self.trainer.train_one_step(0)
        return self.trainer.validate_n_batches(self._budget.eval_n_batches)

    def _after_run(self):
        self._continue_to_full_rate()

        self.adaptation_manager.quantization_rate = 1.0
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)

        lr = self._find_lr()
        self.trainer.train_steps_until_target(
            lr,
            self._budget.max_training_steps,
            self._get_target(),
            0,
            validation_n_batches=self._budget.validation_steps,
            check_interval=self._budget.check_interval,
            patience=3,
        )
        return self.trainer.validate()

    def run(self):
        result = super().run()
        return self.trainer.validate()
