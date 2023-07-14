from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.layers import ClampedReLU

import torch.nn as nn
class PretrainingStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["init_model"]
        promises = ["pretrained_model", "pt_accuracy"]
        clears = ["init_model"]
        super().__init__(requires, promises, clears, pipeline)


    def process(self):
        model = self.pipeline.cache["init_model"]

        trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            self.pipeline.data_provider, 
            self.pipeline.loss)
        trainer.report_function = self.pipeline.reporter.report
        
        model.set_activation(nn.LeakyReLU())
        validation_accuracy = trainer.train_n_epochs(
            self.pipeline.config['lr'], 
            self.pipeline.config['training_epochs'])
        
        
        self.pipeline.cache.add("pretrained_model", model, 'torch_model')
        self.pipeline.cache.add("pt_accuracy", validation_accuracy)

        self.pipeline.cache.remove("init_model")
