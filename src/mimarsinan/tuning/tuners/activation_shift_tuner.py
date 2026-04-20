"""One-shot activation-shift tuner with recovery training.

Applies the full shift once, then recovers accuracy with LR search +
step-budgeted training. Not a smooth adaptation -- extends TunerBase directly.
"""

import torch

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.shift_calculation import calculate_activation_shift
from mimarsinan.tuning.unified_tuner import TunerBase


class ActivationShiftTuner(TunerBase):
    """Apply activation-shift semantics, then recover with step-budgeted training."""

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        self._use_ttfs = pipeline.config.get("spiking_mode", "rate") in (
            "ttfs",
            "ttfs_quantized",
        )
        self._final_metric = None
        self.name = "Shift Recovery"

    def _apply_shift(self):
        config = self.pipeline.config
        transformer = PerceptronTransformer()
        if not self._use_ttfs:
            self.adaptation_manager.shift_rate = 1.0

        for perceptron in self.model.get_perceptrons():
            if not self._use_ttfs:
                shift_amount = calculate_activation_shift(
                    config["target_tq"], perceptron.activation_scale
                )
                act_scale = perceptron.activation_scale
                if torch.is_tensor(act_scale) and torch.is_tensor(shift_amount):
                    act_scale = act_scale.to(shift_amount.device)
                effective_bias_shift = shift_amount / act_scale

                self.adaptation_manager.update_activation(config, perceptron)
                transformer.apply_effective_bias_transform(
                    perceptron,
                    lambda bias, delta=effective_bias_shift: bias + delta,
                )
            else:
                self.adaptation_manager.update_activation(config, perceptron)

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()

    def run(self):
        self._apply_shift()
        self.pipeline.reporter.report(self.name, 1.0)
        self.pipeline.reporter.report("Adaptation target", self._get_target())

        lr = self._find_lr()
        self.trainer.train_steps_until_target(
            lr,
            self._budget.max_training_steps,
            self._get_target(),
            0,
            validation_n_batches=self._budget.progress_eval_batches,
            check_interval=self._budget.check_interval,
            patience=5,
            min_steps=self._budget.check_interval * 3,
            min_improvement=self._budget.accuracy_se(),
        )
        # Tuner internals never call ``trainer.test()`` — the pipeline's
        # ``PipelineStep.pipeline_metric()`` owns the single test() pass.
        self._final_metric = self.trainer.validate()
        return self._final_metric
