"""Activation Adaptation (no-quant): ReLU replacement + scales, no clamp.

Runs when activation_quantization is False. Replaces non-ReLU chip-targeted
activations (e.g. GELU, LeakyReLU) with ReLU via training, applies
activation_scales, and does not set clamp_rate so Normalization Fusion →
Soft Core Mapping stays exact.
"""

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.layers import TransformedActivation
from mimarsinan.pipelining.pipeline_steps.activation_utils import (
    HOST_SIDE_TYPES,
    needs_clamp_adaptation,
    RELU_COMPATIBLE_TYPES,
)
from mimarsinan.models.perceptron_mixer.perceptron import make_activation
from mimarsinan.mapping.mappers.base import resolve_activation_type


class ActivationAdaptationStep(PipelineStep):
    """ReLU adaptation and scale application when activation_quantization is off.

    Does not run ClampAdaptationStep, so clamp_rate stays 0 and the clamp
    decorator remains a no-op. If any chip-targeted perceptron has a non-ReLU
    base (GELU, LeakyReLU), replaces it with ReLU and runs a short adaptation
    phase to recover accuracy. Applies activation_scales to all perceptrons.
    """

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager", "activation_scales"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        if self.trainer is not None:
            return self.trainer.validate()
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        scales = self.get_entry("activation_scales")

        if needs_clamp_adaptation(model):
            # Replace non-ReLU chip-targeted bases with ReLU (no clamp).
            replaced = []
            for p in model.get_perceptrons():
                base = p.base_activation
                name = type(base).__name__
                if name in HOST_SIDE_TYPES:
                    continue
                if name not in RELU_COMPATIBLE_TYPES:
                    old_name = getattr(p, "name", None) or "?"
                    replaced.append((old_name, name))
                    p.base_activation = make_activation("ReLU")
                    p.base_activation_name = "ReLU"
                    p.set_activation(TransformedActivation(p.base_activation, []))

            n_replaced = len(replaced)
            if n_replaced > 0:
                print(
                    f"[ActivationAdaptationStep] Replaced {n_replaced} perceptron(s) "
                    f"from non-ReLU to ReLU (e.g. {replaced[:3]})."
                )
            # Short adaptation phase to recover accuracy after substitution.
            self.trainer = BasicTrainer(
                model,
                self.pipeline.config["device"],
                DataLoaderFactory(self.pipeline.data_provider_factory),
                self.pipeline.loss,
            )
            tuner_epochs = int(self.pipeline.config.get("tuner_epochs", 3))
            for _ in range(tuner_epochs):
                self.trainer.train_one_step(0)
            print(
                "[ActivationAdaptationStep] ReLU replacement done; "
                f"adaptation accuracy: {self.validate():.4f}"
            )
        else:
            print(
                "[ActivationAdaptationStep] All activations are ReLU-compatible — "
                "applying scales only."
            )

        # Apply activation_scales to every perceptron (no clamp_rate change).
        for idx, perceptron in enumerate(model.get_perceptrons()):
            perceptron.set_activation_scale(scales[idx])

        # Verify activation types as the mapper/IR will see them (diagnostic).
        act_types = {}
        for p in model.get_perceptrons():
            t = resolve_activation_type(p)
            base = (t or "").split(" + ")[0].strip()
            act_types[base] = act_types.get(base, 0) + 1
        if act_types:
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(act_types.items()))
            print(f"[ActivationAdaptationStep] Activation types (as seen by IR): {summary}")

        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, "torch_model")
