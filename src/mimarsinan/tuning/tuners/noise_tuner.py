"""Tuner for gradual noise injection."""

from mimarsinan.tuning.adaptation_rate_tuner import AdaptationRateTuner


class NoiseTuner(AdaptationRateTuner):
    rate_attr = "noise_rate"

    def _update_and_evaluate(self, rate):
        self._apply_rate(rate)
        self.trainer.train_one_step(self.lr)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def run(self):
        super().run()
        return self.trainer.validate()
