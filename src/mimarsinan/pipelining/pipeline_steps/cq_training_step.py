from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import NormalizationAwarePerceptronQuantizationTuner
from mimarsinan.models.layers import FrozenStatsNormalization
from mimarsinan.tuning.adaptation_manager import AdaptationManager

import torch.nn as nn

class FullCQNormalizationAwarePerceptronQuantizationTuner(NormalizationAwarePerceptronQuantizationTuner):
    def __init__(self, 
                 pipeline, 
                 model,
                 quantization_bits, 
                 target_tq,
                 target_accuracy,
                 lr):

        super().__init__(pipeline, model, quantization_bits, target_tq, target_accuracy, lr)

    def _get_target_decay(self):
        return 0.99

    def _update_and_evaluate(self, rate):
        adaptation_manager = AdaptationManager()
        adaptation_manager.scale_rate = 0.0
        adaptation_manager.shift_rate = 0.0
        adaptation_manager.quantization_rate = 1.0
        adaptation_manager.clamp_rate = 1.0

        for perceptron in self.model.get_perceptrons():
            perceptron.activation_scale = 1.0
            adaptation_manager.update_activation(self.pipeline.config, perceptron)

        return super()._update_and_evaluate(rate)

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
        return self.tuner.validate()

    def process(self):
        model = self.get_entry("model")

        self.tuner = FullCQNormalizationAwarePerceptronQuantizationTuner(
            self.pipeline,
            model = model,
            quantization_bits = self.pipeline.config['weight_bits'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()

        for perceptron in model.get_perceptrons():
            if not isinstance(perceptron.normalization, nn.Identity):
                for param in perceptron.normalization.parameters():
                    param.requires_grad = False

                perceptron.normalization = \
                    FrozenStatsNormalization(perceptron.normalization)

        self.tuner = FullCQNormalizationAwarePerceptronQuantizationTuner(
            self.pipeline,
            model = model,
            quantization_bits = self.pipeline.config['weight_bits'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        self.tuner.run()
        
        self.update_entry("model", model, 'torch_model')