from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import NormalizationAwarePerceptronQuantizationTuner
from mimarsinan.models.layers import FrozenStatsNormalization

import torch.nn as nn

class WeightQuantizationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None
    
    def validate(self):
        return self.tuner.validate()

    def process(self):
        model = self.get_entry("model")

        for perceptron in model.get_perceptrons():
            if not isinstance(perceptron.normalization, nn.Identity):
                for param in perceptron.normalization.parameters():
                    param.requires_grad = False

                perceptron.normalization = \
                    FrozenStatsNormalization(perceptron.normalization)
                
        self.tuner = NormalizationAwarePerceptronQuantizationTuner(
            self.pipeline,
            model = model,
            quantization_bits = self.pipeline.config['weight_bits'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()
        
        self.update_entry("model", model, 'torch_model')