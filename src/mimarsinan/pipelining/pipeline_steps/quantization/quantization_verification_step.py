import torch

from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


class QuantizationVerificationStep(TrainerPipelineStep):
    REQUIRES = ("model",)

    @classmethod
    def applies_to(cls, plan):
        return plan.weight_quantization

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        self.q_max = (2 ** (self.pipeline.config["weight_bits"] - 1)) - 1

    def process(self):
        model = self.get_entry("model")
        self.trainer = make_basic_trainer(self.pipeline, model)
        for perceptron in model.get_perceptrons():
            print(perceptron.parameter_scale)
            print(perceptron.scale_factor)
            perceptron.to(self.pipeline.config["device"])
            _fused_w = PerceptronTransformer().get_effective_weight(perceptron)
            _fused_b = PerceptronTransformer().get_effective_bias(perceptron)
            param_scale = perceptron.parameter_scale
            assert torch.allclose(
                _fused_w * param_scale,
                torch.round(_fused_w * param_scale),
                atol=1e-3,
                rtol=1e-3,
            ), f"{_fused_w * param_scale}"
            assert torch.allclose(
                _fused_b * param_scale,
                torch.round(_fused_b * param_scale),
                atol=1e-3,
                rtol=1e-3,
            ), f"{_fused_b * param_scale}"
