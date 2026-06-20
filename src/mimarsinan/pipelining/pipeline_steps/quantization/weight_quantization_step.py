from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.layers import FrozenStatsNormalization
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)

import torch.nn as nn


class WeightQuantizationStep(TunerPipelineStep):
    @classmethod
    def applies_to(cls, plan):
        return plan.weight_quantization

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        compute_per_source_scales(model.get_mapper_repr())
        for perceptron in model.get_perceptrons():
            if not isinstance(perceptron.normalization, nn.Identity):
                for param in perceptron.normalization.parameters():
                    param.requires_grad = False
                perceptron.normalization = FrozenStatsNormalization(
                    perceptron.normalization
                )
        bits = self.pipeline.config["weight_bits"]
        print(f"Quantizing to {bits} bits")
        self.run_tuner(
            NormalizationAwarePerceptronQuantizationTuner,
            model,
            adaptation_manager,
            quantization_bits=bits,
        )
