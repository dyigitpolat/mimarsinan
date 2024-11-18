from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.model_training.basic_trainer import BasicTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.layers import SavedTensorDecorator, TransformedActivation

import torch.nn as nn
import torch

class InputActivationAnalysisStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = []
        updates = ["model"]
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

        self.trainer.validate()

        in_scales = [1.0]
        for g_idx, perceptron_group in enumerate(model.perceptron_flow.get_perceptron_groups()):
            total_scale = 0.0
            for perceptron in perceptron_group:
                total_scale += perceptron.activation_scale.item()

            s = total_scale / len(perceptron_group)
            in_scales.append(s)

        for g_idx, perceptron_group in enumerate(model.perceptron_flow.get_perceptron_groups()):
            for perceptron in perceptron_group:
                perceptron.set_input_scale(in_scales[g_idx])
                
        print(in_scales)
        self.update_entry("model", model, 'torch_model')

        