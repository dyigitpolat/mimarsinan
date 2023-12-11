from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch.nn as nn
import torch

class ActivationShiftStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
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
        
        adaptation_manager = self.get_entry('adaptation_manager')
        for perceptron in model.get_perceptrons():
            shift_amount = calculate_activation_shift(self.pipeline.config["target_tq"], perceptron.activation_scale)

            adaptation_manager.shift_rate = 1.0
            adaptation_manager.update_activation(self.pipeline.config, perceptron)

            PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda b: b + shift_amount)
        
        
        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, 'torch_model')
