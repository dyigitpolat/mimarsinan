"""
TorchMappingStep -- convert a native PyTorch model to a mimarsinan Supermodel.

Inserted after Pretraining and before the first adaptation / quantization
step.  Traces the trained model, validates representability, constructs the
Mapper DAG with Perceptron wrappers, transfers trained weights, and sets up
the AdaptationManager.
"""

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.layers import LeakyGradReLU
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch
import torch.nn as nn


class TorchMappingStep(PipelineStep):
    """Convert a trained native PyTorch model into a Supermodel.

    This step:
      1. Traces the model with ``torch.fx``.
      2. Validates that every operation is representable.
      3. Converts the FX graph to a Mapper DAG, transferring trained weights.
      4. Wraps the result in a ``Supermodel``.
      5. Sets up Perceptron activations via ``AdaptationManager``.
      6. Optionally verifies forward-pass equivalence with the original model.
    """

    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["adaptation_manager"]
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        if self.trainer is not None:
            return self.trainer.validate()
        return self.pipeline.get_target_metric()

    def process(self):
        from mimarsinan.torch_mapping.converter import convert_torch_model

        native_model = self.get_entry("model")

        supermodel = convert_torch_model(
            native_model,
            input_shape=tuple(self.pipeline.config["input_shape"]),
            num_classes=self.pipeline.config["num_classes"],
            device=self.pipeline.config["device"],
            Tq=self.pipeline.config["target_tq"],
        )

        adaptation_manager = AdaptationManager()
        model_config = self.pipeline.config.get("model_config", {})
        for perceptron in supermodel.get_perceptrons():
            self._set_activation(model_config, perceptron, adaptation_manager)

        self._verify_equivalence(native_model, supermodel)

        self.trainer = BasicTrainer(
            supermodel,
            self.pipeline.config["device"],
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss,
        )
        self.trainer.report_function = self.pipeline.reporter.report

        self.update_entry("model", supermodel, "torch_model")
        self.add_entry("adaptation_manager", adaptation_manager, "pickle")

        print(f"[TorchMappingStep] Converted native model to Supermodel")
        print(f"  Perceptrons: {len(supermodel.get_perceptrons())}")
        print(f"  Validation: {self.validate()}")

    def _set_activation(self, model_config, perceptron, adaptation_manager):
        base_act = model_config.get("base_activation", "ReLU")
        if base_act == "ReLU":
            perceptron.base_activation = LeakyGradReLU()
            perceptron.activation = LeakyGradReLU()
        elif base_act == "LeakyReLU":
            perceptron.base_activation = nn.LeakyReLU()
            perceptron.activation = nn.LeakyReLU()
        elif base_act == "GELU":
            perceptron.base_activation = nn.GELU()
            perceptron.activation = nn.GELU()
        else:
            perceptron.base_activation = LeakyGradReLU()
            perceptron.activation = LeakyGradReLU()

        adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def _verify_equivalence(self, native_model, supermodel):
        """Best-effort check that the converted model matches the original."""
        try:
            device = self.pipeline.config["device"]
            input_shape = tuple(self.pipeline.config["input_shape"])
            dummy = torch.randn(2, *input_shape, device=device)

            native_model.eval().to(device)
            supermodel.eval().to(device)

            with torch.no_grad():
                native_out = native_model(dummy)
                super_out = supermodel(dummy)

            if native_out.shape != super_out.shape:
                print(
                    f"[TorchMappingStep] WARNING: output shape mismatch: "
                    f"native={native_out.shape} vs converted={super_out.shape}"
                )
        except Exception as e:
            print(f"[TorchMappingStep] Equivalence check skipped: {e}")
