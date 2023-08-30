from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import NormalizationAwarePerceptronQuantizationTuner

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
        self.tuner = NormalizationAwarePerceptronQuantizationTuner(
            self.pipeline,
            model = self.get_entry("model"),
            quantization_bits = self.pipeline.config['weight_bits'],
            target_tq = self.pipeline.config['target_tq'],
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'])
        #self.tuner.run()

        from mimarsinan.transformations.normalization_aware_perceptron_quantization import NormalizationAwarePerceptronQuantization
        for perceptron in self.get_entry("model").get_perceptrons():
            perceptron.layer = NormalizationAwarePerceptronQuantization(
                self.pipeline.config['weight_bits'], self.pipeline.config['device']).transform(perceptron).layer

        from mimarsinan.mapping.mapping_utils import get_fused_weights
        import torch
        import torch.nn as nn
        for perceptron in self.get_entry("model").get_perceptrons():
            if not isinstance(perceptron.normalization, nn.Identity):
                assert isinstance(perceptron.normalization, nn.BatchNorm1d)
                fused_w, fused_b = get_fused_weights(perceptron.layer, perceptron.normalization)
                
                w_max = torch.max(torch.abs(fused_w))
                b_max = torch.max(torch.abs(fused_b))
                scale_param = 7 / max(w_max, b_max)
                
                assert torch.allclose(fused_w * scale_param, torch.round(fused_w * scale_param),
                                      rtol=1e-03, atol=1e-03)
                assert torch.allclose(fused_b * scale_param, torch.round(fused_b * scale_param),
                                      rtol=1e-03, atol=1e-03)
                print ("verified bn")
            else:
                if perceptron.layer.bias is None:
                    max_w = perceptron.layer.weight.data.abs().max()
                    scale_param = 7 / max_w
                    assert torch.allclose(perceptron.layer.weight.data * scale_param, torch.round(perceptron.layer.weight.data * scale_param),
                                          rtol=1e-03, atol=1e-03)
                else:
                    max_w = perceptron.layer.weight.data.abs().max()
                    max_b = perceptron.layer.bias.data.abs().max()
                    scale_param = 7 / max(max_w, max_b)
                    assert torch.allclose(perceptron.layer.weight.data * scale_param, torch.round(perceptron.layer.weight.data * scale_param),
                                          rtol=1e-03, atol=1e-03)
                    assert torch.allclose(perceptron.layer.bias.data * scale_param, torch.round(perceptron.layer.bias.data * scale_param),
                                          rtol=1e-03, atol=1e-03)
                print ("verified non")

        self.update_entry("model", self.tuner.model, 'torch_model')