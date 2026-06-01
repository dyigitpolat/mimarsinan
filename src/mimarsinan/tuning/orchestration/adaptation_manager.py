from mimarsinan.models.nn.layers import *
from mimarsinan.tuning.shift_calculation import calculate_activation_shift

import torch.nn as nn

class AdaptationManager(nn.Module):
    def __init__(self):
        super(AdaptationManager, self).__init__()

        self.activation_adaptation_rate = 0.0
        self.clamp_rate = 0.0
        self.shift_rate = 0.0
        self.quantization_rate = 0.0
        self.scale_rate = 0.0
        self.pruning_rate = 0.0

        self.noise_rate = 0.0

        # When True, the on-chip activation already applies clamp/quant/shift
        # internally (LIFActivation / TTFSCycleActivation) — skip duplicate decorators.
        self.lif_active = False
        self.ttfs_active = False

    def update_activation(self, pipeline_config, perceptron):
        from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing

        spiking_mode = pipeline_config.get("spiking_mode", "lif")
        use_ttfs = requires_ttfs_firing(spiking_mode)
        subsumes_decorators = (
            getattr(self, "lif_active", False)
            or getattr(self, "ttfs_active", False)
            or spiking_mode == "lif"
        )
        decorators = []
        if self.activation_adaptation_rate > 0:
            decorators.append(
                self.get_rate_adjusted_activation_replacement_decorator(perceptron))
        if not subsumes_decorators and self.clamp_rate != 0.0:
            decorators.append(self.get_rate_adjusted_clamp_decorator(perceptron))
        if not subsumes_decorators and self.quantization_rate != 0.0:
            decorators.append(
                self.get_rate_adjusted_quantization_decorator(pipeline_config, perceptron))
        if not subsumes_decorators and not use_ttfs and self.shift_rate != 0.0:
            decorators.append(self.get_shift_decorator(pipeline_config, perceptron))

        perceptron.set_activation(
            TransformedActivation(perceptron.base_activation, decorators))

        if self.noise_rate > 0:
            target_noise_amount = (1.0 / (pipeline_config['target_tq'] * 2.5))
            perceptron.set_regularization(
                NoisyDropout(torch.tensor(0.0), self.noise_rate, target_noise_amount * perceptron.activation_scale)
            )
        else:
            perceptron.set_regularization(nn.Identity())

    def get_rate_adjusted_activation_replacement_decorator(self, perceptron):
        """Gradually blend the base activation toward LeakyGradReLU (chip ReLU)."""
        from mimarsinan.models.nn.activations import LeakyGradReLU
        return RateAdjustedDecorator(
            self.activation_adaptation_rate,
            ActivationReplacementDecorator(LeakyGradReLU()),
            MixAdjustmentStrategy())

    def get_rate_adjusted_clamp_decorator(self, perceptron):
        return RateAdjustedDecorator(
            self.clamp_rate, 
            ClampDecorator(torch.tensor(0.0), perceptron.activation_scale), 
            MixAdjustmentStrategy())

    def get_shift_decorator(self, pipeline_config, perceptron):
        shift_amount = calculate_activation_shift(pipeline_config["target_tq"], perceptron.activation_scale) * self.shift_rate
        return ShiftDecorator(shift_amount)
    
    def get_rate_adjusted_quantization_decorator(self, pipeline_config, perceptron):
        from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing

        use_ttfs = requires_ttfs_firing(pipeline_config.get("spiking_mode", "lif"))
        shift = calculate_activation_shift(
            pipeline_config["target_tq"], perceptron.activation_scale
        )
        if use_ttfs:
            shift_back_amount = -shift
        else:
            shift_back_amount = -shift * self.shift_rate

        return RateAdjustedDecorator(
            self.quantization_rate, 
            NestedDecoration(
                [ShiftDecorator(shift_back_amount), 
                QuantizeDecorator(torch.tensor(pipeline_config["target_tq"]), perceptron.activation_scale)]),
            NestedAdjustmentStrategy([RandomMaskAdjustmentStrategy(), MixAdjustmentStrategy()]))
    