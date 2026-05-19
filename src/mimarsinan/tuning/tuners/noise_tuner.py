"""Tuner for gradual noise injection."""

from mimarsinan.tuning.adaptation_rate_tuner import AdaptationRateTuner


class NoiseTuner(AdaptationRateTuner):
    rate_attr = "noise_rate"

    def _update_and_evaluate(self, rate):
        setattr(self.adaptation_manager, self.rate_attr, rate)
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        self.trainer.train_one_step(self.lr)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _after_run(self):
        super()._after_run()
        return self._ensure_pipeline_threshold()

    def run(self):
        super().run()
        return self.trainer.validate()
