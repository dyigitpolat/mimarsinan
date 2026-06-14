"""Weight quantization tuner using NormalizationAwarePerceptronQuantization."""

import torch

from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.tuning.tuners.perceptron_transform_tuner import PerceptronTransformTuner


class NormalizationAwarePerceptronQuantizationTuner(PerceptronTransformTuner):
    def __init__(self, pipeline, model, quantization_bits, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.quantization_bits = quantization_bits
        self.adaptation_manager = adaptation_manager
        self._axis = None
        if pipeline.config.get("tuning_use_axis", False):
            from mimarsinan.tuning.axes import NAPQAxis

            self._axis = NAPQAxis(self._apply_rate)
            self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)

    def _get_previous_perceptron_transform(self, rate):
        return lambda perceptron: None

    def _get_new_perceptron_transform(self, rate):
        def transform(perceptron):
            NormalizationAwarePerceptronQuantization(
                self.quantization_bits,
                self.pipeline.config["device"],
                rate,
            ).transform(perceptron)
        return transform

    def _apply_rate(self, rate):
        self.trainer.perceptron_transformation = self._mixed_transform(rate)
        with torch.no_grad():
            self.trainer._update_and_transform_model()

    def _update_and_evaluate(self, rate):
        if getattr(self, "_axis", None) is not None:
            self._axis.set_rate(rate)
        else:
            self._apply_rate(rate)
        return self.trainer.validate_n_batches(self._budget.eval_n_batches)

    def run(self):
        return super().run()
