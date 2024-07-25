from mimarsinan.models.perceptron_mixer.simple_mlp import SimpleMLP
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.layers import LeakyGradReLU
from mimarsinan.models.preprocessing.input_cq import InputCQ
from mimarsinan.models.preprocessing.channel_expansion import ChannelExpansion

import torch.nn as nn

class SimpleMLPBuilder:
    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        preprocessor = InputCQ(self.pipeline_config["target_tq"])

        perceptron_flow = SimpleMLP(self.device, self.input_shape, self.num_classes, configuration['mlp_width_1'], configuration['mlp_width_2'])
        adaptation_manager = AdaptationManager()
        supermodel = Supermodel(self.device, self.input_shape, self.num_classes, preprocessor, perceptron_flow, self.pipeline_config["target_tq"])
        for perceptron in supermodel.get_perceptrons():
            perceptron.base_activation = LeakyGradReLU()
            adaptation_manager.update_activation(self.pipeline_config, perceptron)

            assert perceptron.layer.weight.shape[0] <= self.max_neurons, f"not enough neurons ({perceptron.layer.weight.shape[0]} > {self.max_neurons})"
            assert perceptron.layer.weight.shape[1] <= self.max_axons - 1, f"not enough axons ({perceptron.layer.weight.shape[1]} > {self.max_axons})"

        return supermodel