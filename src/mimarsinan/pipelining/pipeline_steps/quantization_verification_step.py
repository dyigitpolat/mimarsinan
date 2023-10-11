from mimarsinan.mapping.mapping_utils import get_fused_weights
import torch
import torch.nn as nn
        

from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.models.layers import FrozenStatsNormalization

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
            assert isinstance(perceptron.layer, nn.Linear) or isinstance(perceptron.layer, nn.Identity)

            if not isinstance(perceptron.normalization, nn.Identity):
                print("bn")

                assert isinstance(perceptron.normalization, FrozenStatsNormalization)
                fused_w, fused_b = get_fused_weights(perceptron.layer, perceptron.normalization)
                
                scale_param = self.q_max
                
                print(fused_w * scale_param)
                assert torch.allclose(fused_w * scale_param, torch.round(fused_w * scale_param),
                                      rtol=1e-02, atol=1e-02)
                assert torch.allclose(fused_b * scale_param, torch.round(fused_b * scale_param),
                                      rtol=1e-02, atol=1e-02)
                print ("verified bn")
            else:
                if perceptron.layer.bias is None:
                    print("no bn, no bias")
                    
                    scale_param = self.q_max
                    assert torch.allclose(perceptron.layer.weight.data * scale_param, torch.round(perceptron.layer.weight.data * scale_param),
                                          rtol=1e-03, atol=1e-03)
                else:
                    print("no bn")
                    scale_param = self.q_max
                    q_w = perceptron.layer.weight.data * scale_param
                    q_b = perceptron.layer.bias.data * scale_param

                    assert torch.allclose(q_w, torch.round(q_w),
                                          rtol=1e-03, atol=1e-03)
                    assert torch.allclose(q_b, torch.round(q_b),
                                          rtol=1e-03, atol=1e-03)
                print ("verified non")