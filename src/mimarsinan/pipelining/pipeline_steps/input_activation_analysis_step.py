from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.model_training.basic_trainer import BasicTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.layers import SavedTensorDecorator, TransformedActivation

import torch.nn as nn
import torch

class InputActivationAnalysisStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["input_activation_scales", "output_activation_scales"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")

        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)

        for perceptron in model.get_perceptrons():
            perceptron.input_activation = TransformedActivation(nn.Identity(), [SavedTensorDecorator()])
            perceptron.activation.decorate(SavedTensorDecorator())

        self.trainer.validate()

        activation_scales = []
        output_activation_scales = []
        for perceptron in model.get_perceptrons():
            saved_tensor = perceptron.input_activation.pop_decorator()
            saved_tensor_out = perceptron.activation.pop_decorator()
            activation_scales.append(torch.max(saved_tensor.latest_input.view(-1)).item())
            output_activation_scales.append(torch.max(saved_tensor_out.latest_input.view(-1)).item())

        print(activation_scales)

        self.add_entry("input_activation_scales", activation_scales)
        self.add_entry("output_activation_scales", output_activation_scales)


        