from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.mapping.support.bias_compensation import (
    apply_lif_half_step_bias_compensation,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.layers import FrozenStatsNormalization
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)

import torch.nn as nn


class WeightQuantizationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    @classmethod
    def applies_to(cls, plan):
        return plan.weight_quantization

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        self._apply_lif_half_step_entry_fold(model)
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

    def _apply_lif_half_step_entry_fold(self, model) -> None:
        """[5v B3] Fold the LIF theta/(2T) half-step as a TRAINABLE entry bias
        BEFORE the weight-quantization QAT, so the QAT reconciles the shifted
        operating point and the float NF stays bit-exact with the quantized
        deployed sim. Injected post-QAT at soft-core mapping it broke the LIF
        parity identity (t0_01 0.9336, t0_05 0.9883). Recipe-knob-gated on
        lif x activation_quantization; idempotent per perceptron."""
        plan = DeploymentPlan.of(self.pipeline)
        if not is_lif(plan.spiking_mode) or not plan.activation_quantization:
            return
        if not bool(self.pipeline.config.get("lif_half_step_bias", False)):
            return
        folded = apply_lif_half_step_bias_compensation(
            model, int(self.pipeline.config["simulation_steps"]),
        )
        print(
            f"[WeightQuantizationStep] LIF half-step head-start folded on "
            f"{folded} perceptrons before the QAT (theta/(2T), T="
            f"{int(self.pipeline.config['simulation_steps'])})."
        )
