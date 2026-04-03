"""Tuner for gradual activation adaptation (non-ReLU -> ReLU).

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

from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class ActivationAdaptationTuner(SmoothAdaptationTuner):
    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager

        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def _get_extra_state(self):
        return self.adaptation_manager.activation_adaptation_rate

    def _set_extra_state(self, extra):
        self.adaptation_manager.activation_adaptation_rate = extra
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)

    def _update_and_evaluate(self, rate):
        self.adaptation_manager.activation_adaptation_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        self.trainer.train_one_step(0)
        return self.trainer.validate_n_batches(self._budget.eval_n_batches)

    def _after_run(self):
        from mimarsinan.models.perceptron_mixer.perceptron import make_activation
        from mimarsinan.pipelining.pipeline_steps.activation_utils import (
            needs_relu_adaptation,
        )

        pre_commit_state = self._clone_state()
        pre_commit_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)

        for p in self.model.get_perceptrons():
            if needs_relu_adaptation(p):
                p.base_activation = make_activation("ReLU")
                p.base_activation_name = "ReLU"

        self.adaptation_manager.activation_adaptation_rate = 0.0
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        post_commit_acc = self.trainer.test()

        if post_commit_acc < self.target_adjuster.floor:
            self._restore_state(pre_commit_state)
            self._committed_metric = pre_commit_acc
        else:
            self._committed_metric = post_commit_acc

        return self._committed_metric

    def validate(self):
        if hasattr(self, "_committed_metric") and self._committed_metric is not None:
            return self._committed_metric
        return self.trainer.validate()
