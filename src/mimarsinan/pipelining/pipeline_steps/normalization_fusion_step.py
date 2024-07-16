from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator, ScaleDecorator

from mimarsinan.visualization.activation_function_visualization import ActivationFunctionVisualizer

import torch.nn as nn
import torch

class NormalizationFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model"]
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
        
        for perceptron in model.get_perceptrons():
            print(perceptron.parameter_scale)
            print(perceptron.scale_factor)

            perceptron.to(self.pipeline.config['device'])
            w = PerceptronTransformer().get_effective_weight(perceptron)
            b = PerceptronTransformer().get_effective_bias(perceptron)

            perceptron.layer = nn.Linear(
                perceptron.input_features, 
                perceptron.output_channels, bias=True)
            
            perceptron.layer.weight.data = w 
            perceptron.layer.bias.data = b 

            perceptron.normalization = nn.Identity()

            perceptron.set_activation_scale(1.0)
            perceptron.set_input_scale(1.0)
            adaptation_manager.update_activation(self.pipeline.config, perceptron)
        
        print(self.validate())

        self.update_entry("model", model, 'torch_model')