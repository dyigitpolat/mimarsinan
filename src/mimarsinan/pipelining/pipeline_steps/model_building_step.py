from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.layers import LeakyGradReLU

import torch.nn as nn
import torch

class ModelBuildingStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model_config", "model_builder"]
        promises = ["model", "adaptation_manager"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()
    
    def set_activation(self, model_config, perceptron, adaptation_manager):
        if model_config['base_activation'] == "ReLU":
            perceptron.base_activation = LeakyGradReLU()
            perceptron.activation = LeakyGradReLU()
        elif model_config['base_activation'] == "LeakyReLU":
            perceptron.base_activation = nn.LeakyReLU()
            perceptron.activation = nn.LeakyReLU()
        elif model_config['base_activation'] == "GELU":
            perceptron.base_activation = nn.GELU()
            perceptron.activation = nn.GELU()
        else:
            print("No (or Invalid) base activation provided, using LeakyGradReLU")
            perceptron.base_activation = LeakyGradReLU()
            perceptron.activation = LeakyGradReLU()
        
        adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def _is_supermodel(self, model):
        return hasattr(model, "get_perceptrons") and hasattr(model, "get_mapper_repr")

    def process(self):
        builder = self.get_entry('model_builder')
        init_model = builder.build(self.get_entry("model_config"))

        adaptation_manager = AdaptationManager()

        if self._is_supermodel(init_model):
            for perceptron in init_model.get_perceptrons():
                self.set_activation(self.get_entry("model_config"), perceptron, adaptation_manager)

        # Warmup forward pass to initialize any Lazy modules (e.g. LazyBatchNorm1d),
        # so subsequent transformations / mapping that touch normalization parameters
        # won't crash if the pipeline is resumed from a later step.
        try:
            init_model.eval()
            with torch.no_grad():
                input_shape = tuple(self.pipeline.config["input_shape"])
                dummy = torch.zeros((1, *input_shape))
                _ = init_model(dummy)
        except Exception as e:
            print(f"[ModelBuildingStep] Warmup forward failed: {e}")

        self.add_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.add_entry("model", (init_model), "torch_model")