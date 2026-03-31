"""Tuner for gradual activation adaptation (non-ReLU → ReLU).

Uses SmartSmoothAdaptation to progressively blend from the original
activation (e.g. GELU, LeakyReLU) to ReLU, following the same pattern
as ClampTuner and ActivationQuantizationTuner.

Does not apply activation_scales -- that is the responsibility of
downstream steps (Clamp Adaptation, etc.).

Contract: ``model.get_perceptrons()`` only returns chip-targeted perceptrons
(Identity host-side perceptrons are excluded by the mapper's
``owned_perceptron_groups()`` implementation).  This tuner therefore does not
need to special-case Identity; ``needs_relu_adaptation`` handles only the
already-ReLU-compatible check.
"""

from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner


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
        from mimarsinan.pipelining.pipeline_steps.activation_utils import (
            needs_relu_adaptation,
        )

        for p in self.model.get_perceptrons():
            if needs_relu_adaptation(p):
                p.base_activation = make_activation("ReLU")
                p.base_activation_name = "ReLU"
        # Reset the rate so the decorator no longer applies.
        self.adaptation_manager.activation_adaptation_rate = 0.0
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        # Measure accuracy on the full test set after the commit.
        # trainer.validate() evaluates only one minibatch, which introduces
        # significant batch-to-batch noise (observed swings from 0.95 to 0.63
        # on MNIST).  That noisy value would propagate as the recovery TARGET
        # for ClampTuner, causing it to aim for 0.63 instead of 0.96.
        # trainer.test() iterates the complete test split, giving a stable,
        # reliable metric that both the pipeline tolerance check and downstream
        # tuners can use as a ground-truth accuracy baseline.
        self._committed_metric = self.trainer.test()
        return self._committed_metric
