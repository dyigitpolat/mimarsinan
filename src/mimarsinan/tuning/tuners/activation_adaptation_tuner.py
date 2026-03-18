"""Tuner for gradual activation adaptation (non-ReLU → ReLU).

Uses SmartSmoothAdaptation to progressively blend from the original
activation (e.g. GELU, LeakyReLU) to ReLU, following the same pattern
as ClampTuner and ActivationQuantizationTuner.

Does not apply activation_scales -- that is the responsibility of
downstream steps (Clamp Adaptation, etc.).
"""

from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner
from mimarsinan.pipelining.pipeline_steps.activation_utils import RELU_COMPATIBLE_TYPES


class ActivationAdaptationTuner(PerceptronTuner):
    def __init__(
        self,
        pipeline,
        model,
        target_accuracy,
        lr,
        adaptation_manager,
    ):
        super().__init__(pipeline, model, target_accuracy, lr)

        self.adaptation_manager = adaptation_manager

        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def _get_target_decay(self):
        return 0.99

    def _update_and_evaluate(self, rate):
        self.adaptation_manager.activation_adaptation_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.trainer.train_one_step(0)
        return self.trainer.validate()

    def run(self):
        self.trainer.validate()
        super().run()

        from mimarsinan.models.perceptron_mixer.perceptron import make_activation

        for p in self.model.get_perceptrons():
            name = type(p.base_activation).__name__
            if name not in RELU_COMPATIBLE_TYPES:
                p.base_activation = make_activation("ReLU")
                p.base_activation_name = "ReLU"
        # Reset the rate so the decorator no longer applies.
        self.adaptation_manager.activation_adaptation_rate = 0.0
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        return self.trainer.validate()
