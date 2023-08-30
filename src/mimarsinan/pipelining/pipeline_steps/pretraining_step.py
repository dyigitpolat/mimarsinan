from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.layers import ClampedReLU

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch.nn as nn
class PretrainingStep(PipelineStep):
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
        
        for perceptron in model.get_perceptrons():
            perceptron.set_activation(nn.LeakyReLU())

        self.trainer.train_n_epochs(
            self.pipeline.config['lr'], 
            self.pipeline.config['training_epochs'])
        
        self.update_entry("model", model, 'torch_model')
