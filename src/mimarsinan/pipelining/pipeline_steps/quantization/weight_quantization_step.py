from typing import Iterable, cast

import torch

from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.mapping.support.bias_compensation import (
    apply_lif_half_step_bias_compensation,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    resolve_wq_two_scale_projection as resolve_wq_two_scale_projection,
)
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.layers import FrozenStatsNormalization
from mimarsinan.transformations.perceptron.bias_canonicalization import (
    canonicalize_starved_bias_outliers,
)
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)

import torch.nn as nn

_CANONICALIZATION_BATCHES = 4


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
        self._canonicalize_starved_bias_outliers(model)
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
        two_scale = resolve_wq_two_scale_projection(self.pipeline.config)
        if bool(self.pipeline.config.get("wq_two_scale_projection", False)) and not two_scale:
            print(
                "[WeightQuantizationStep] wq_two_scale_projection requested but "
                "the platform has no on-chip bias register (param-encoded bias "
                "rows share the weight grid); using the shared-grid projection."
            )
        if two_scale:
            print("[WeightQuantizationStep] two-scale projection: weight grid "
                  "from max|w| alone; bias on its own grid (integer-ratio-snapped).")
        self.run_tuner(
            NormalizationAwarePerceptronQuantizationTuner,
            model,
            adaptation_manager,
            quantization_bits=bits,
            two_scale_projection=two_scale,
        )

    def _apply_lif_half_step_entry_fold(self, model) -> None:
        """Fold the LIF theta/(2T) half-step as a TRAINABLE entry bias BEFORE the
        weight-quantization QAT, so the QAT reconciles the shifted operating point
        and the float NF stays bit-exact with the quantized deployed sim. Idempotent per perceptron."""
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

    def _canonicalize_starved_bias_outliers(self, model) -> None:
        """Guarded empirical bias canonicalization at the QAT entry: outlier bias
        mass the provable OFF-clip cannot reach (empirically constant channels) is
        shrunk to its observed saturation slack and VERIFIED (decision agreement
        on the calibration batches; restored on any flip)."""
        trainer = make_basic_trainer(self.pipeline, model)
        device = self.pipeline.config["device"]
        val_batches = cast(
            "Iterable[tuple[torch.Tensor, torch.Tensor]]",
            trainer.iter_validation_batches(_CANONICALIZATION_BATCHES),
        )
        batches = [x.to(device) for x, _ in val_batches]
        report = canonicalize_starved_bias_outliers(
            model, batches, bits=int(self.pipeline.config["weight_bits"]),
        )
        if any(report.values()):
            self.pipeline.reporter.report(
                "wq_bias_canonicalization", dict(report),
            )
