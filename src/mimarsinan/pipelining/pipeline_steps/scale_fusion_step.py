from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator, ScaleDecorator

from mimarsinan.visualization.activation_function_visualization import ActivationFunctionVisualizer


import torch.nn as nn
import torch

class ScaleFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "activation_scales", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")

        # Trainer
        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report
        
        scale = max(self.get_entry('activation_scales'))
        model.in_act = TransformedActivation(
            base_activation = nn.Identity(),
            decorators = [
                ScaleDecorator(torch.tensor(1.0/scale)),
                ClampDecorator(torch.tensor(0.0), torch.tensor(1.0/scale)),
                QuantizeDecorator(torch.tensor(self.pipeline.config['target_tq']), torch.tensor(1.0/scale)),
            ])
        
        for perceptron in model.get_perceptrons():
            perceptron.set_scale_factor(1.0)
            perceptron.set_activation_scale(1.0)
            adaptation_manager.update_activation(self.pipeline.config, perceptron)
            PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda b: b / scale)

        print(self.validate())

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", model, 'torch_model')