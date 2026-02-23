from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch.nn as nn
import torch

class NormalizationFusionStep(PipelineStep):
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
        self.trainer.report_function = self.pipeline.reporter.report
        
        pt = PerceptronTransformer()
        for perceptron in model.get_perceptrons():
            if isinstance(perceptron.normalization, nn.Identity):
                continue

            perceptron.to(self.pipeline.config['device'])

            u, beta, mean = pt._get_u_beta_mean(perceptron.normalization)

            W = perceptron.layer.weight.data
            b = perceptron.layer.bias.data if perceptron.layer.bias is not None else torch.zeros(W.shape[0], device=W.device)

            fused_W = W * u.unsqueeze(-1)
            fused_b = (b - mean) * u + beta

            perceptron.layer = nn.Linear(
                perceptron.input_features, 
                perceptron.output_channels, bias=True)
            
            perceptron.layer.weight.data = fused_W
            perceptron.layer.bias.data = fused_b

            perceptron.normalization = nn.Identity()
        
        print(self.validate())

        self.update_entry("model", model, 'torch_model')