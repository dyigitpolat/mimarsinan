from mimarsinan.models.perceptron_mixer.perceptron_mixer import PerceptronMixer
from mimarsinan.models.perceptron_mixer.simple_mlp import SimpleMLP
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, LeakyGradReLU

import torch.nn as nn
import torch
class InputCQ(nn.Module):
    def __init__(self, device, input_shape):
        super(InputCQ, self).__init__()
        self.in_act = TransformedActivation(
            base_activation = nn.Identity(),
            decorators = [
                ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))
            ])

    def forward(self, x):
        x = self.in_act(x)
        return x
    
class SimpleMLPBuilder:
    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        preprocessor = InputCQ(self.device, self.input_shape)

        perceptron_flow = SimpleMLP(self.device, self.input_shape, self.num_classes, configuration['mlp_width'], configuration['extra_layers'])
        adaptation_manager = AdaptationManager()
        supermodel = Supermodel(self.device, self.input_shape, self.num_classes, preprocessor, perceptron_flow, self.pipeline_config["target_tq"])
        for perceptron in supermodel.get_perceptrons():
            perceptron.base_activation = LeakyGradReLU()
            #perceptron.base_activation = nn.LeakyReLU()
            adaptation_manager.update_activation(self.pipeline_config, perceptron)

            assert perceptron.layer.weight.shape[0] <= self.max_neurons, f"not enough neurons ({perceptron.layer.weight.shape[0]} > {self.max_neurons})"
            assert perceptron.layer.weight.shape[1] <= self.max_axons - 1, f"not enough axons ({perceptron.layer.weight.shape[1]} > {self.max_axons})"

        return supermodel