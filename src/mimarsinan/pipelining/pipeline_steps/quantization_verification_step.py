import torch
import torch.nn as nn
        

from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

class QuantizationVerificationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = []
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None
        self.q_max = ( 2 ** (self.pipeline.config['weight_bits'] - 1) ) - 1
    
    def validate(self):
        return self.trainer.validate()

    def process(self):
        self.trainer = BasicTrainer(
            self.get_entry("model"), 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report

        for perceptron in self.get_entry("model").get_perceptrons():
            perceptron.to(self.pipeline.config['device'])

            _fused_w = PerceptronTransformer().get_effective_weight(perceptron)
            _fused_b = PerceptronTransformer().get_effective_bias(perceptron)

            param_scale = self.q_max# / p_max

            assert torch.allclose(
                _fused_w * param_scale, torch.round(_fused_w * param_scale),
                atol=1e-3, rtol=1e-3), f"{_fused_w * param_scale}"

            assert torch.allclose(
                _fused_b * param_scale, torch.round(_fused_b * param_scale),
                atol=1e-3, rtol=1e-3), f"{_fused_b * param_scale}"