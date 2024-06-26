from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.adaptation_manager import AdaptationManager

import torch
import torch.nn as nn

class CQTrainingStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None
        self.trainer = None
    
    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")

        adaptation_manager = AdaptationManager()
        adaptation_manager.scale_rate = 0.0
        adaptation_manager.shift_rate = 0.0
        adaptation_manager.noise_rate = 1.0
        adaptation_manager.quantization_rate = 1.0
        adaptation_manager.clamp_rate = 1.0

        for perceptron in model.get_perceptrons():
            perceptron.set_activation_scale(1.0)
            adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report

        self.trainer.train_n_epochs(
            self.pipeline.config['lr'], 
            self.pipeline.config['training_epochs'],
            warmup_epochs=5)
        
        self.update_entry("model", model, 'torch_model')