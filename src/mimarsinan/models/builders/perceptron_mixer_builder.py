from mimarsinan.models.perceptron_mixer.perceptron_mixer import PerceptronMixer
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, LeakyGradReLU

import torch.nn as nn
import torch
class Transduction(nn.Module):
    def __init__(self, device, input_shape):
        super(Transduction, self).__init__()

        self.input_bn = nn.BatchNorm2d(input_shape[-3], device=device)
        self.conv1 = nn.Conv2d(input_shape[-3], input_shape[-3], kernel_size=3, stride=1, padding=1, device=device)
        self.conv2 = nn.Conv2d(input_shape[-3], input_shape[-3], kernel_size=3, stride=1, padding=1, device=device)
        self.in_act = TransformedActivation(
            base_activation = nn.Identity(),
            decorators = [
                ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))
            ])

    def forward(self, x):
        # x = self.input_bn(x)
        # x = self.conv1(x)
        # x = nn.GELU()(x)
        # x = self.conv2(x)
        # x = nn.GELU()(x)
        
        x = self.in_act(x)
        return x
    
class PerceptronMixerBuilder:
    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def validate(self, configuration):
        patch_n_1 = configuration["patch_n_1"]
        patch_m_1 = configuration["patch_m_1"]
        patch_c_1 = configuration["patch_c_1"]
        fc_k_1 = configuration["fc_k_1"]
        fc_w_1 = configuration["fc_w_1"]

        patch_n_2 = configuration["patch_n_2"]
        patch_c_2 = configuration["patch_c_2"]
        fc_k_2 = configuration["fc_k_2"]
        fc_w_2 = configuration["fc_w_2"]

        assert self.input_shape[-2] % patch_n_1 == 0, \
            "mod div != 0 (height)"
        assert self.input_shape[-1] % patch_m_1 == 0, \
            "mod div != 0 (width)"
        
        patch_count = patch_n_1 * patch_m_1
        fc_width = fc_w_1
        patch_height = self.input_shape[-2] // patch_n_1
        patch_width = self.input_shape[-1] // patch_m_1
        input_channels = self.input_shape[-3]
        patch_size = patch_height * patch_width * input_channels

        assert fc_width <= self.max_neurons, f"not enough neurons ({fc_width} > {self.max_neurons})"
        assert fc_width <= self.max_axons - 1, f"not enough axons ({fc_width} > {self.max_axons})"

        assert patch_count <= self.max_neurons, f"not enough neurons ({patch_count} > {self.max_axons})"
        assert patch_count <= self.max_axons - 1, f"not enough axons ({patch_count} > {self.max_axons})"

        assert patch_size <= self.max_axons - 1, f"not enough axons ({patch_size} > {self.max_axons})"

        assert patch_c_1 <= self.max_neurons, f"not enough neurons ({patch_c_1} > {self.max_neurons})"
        assert patch_c_1 <= self.max_axons - 1, f"not enough axons ({patch_c_1} > {self.max_axons})"

        fc_width_2 = fc_w_2
        assert fc_width_2 <= self.max_neurons, f"not enough neurons ({fc_width_2} > {self.max_neurons})"
        assert fc_width_2 <= self.max_axons - 1, f"not enough axons ({fc_width_2} > {self.max_axons})"

        classifier_in = patch_c_1 * patch_count
        assert classifier_in <= self.max_axons - 1, f"not enough axons ({classifier_in} > {self.max_axons})"

    def build(self, configuration):
        patch_n_1 = configuration["patch_n_1"]
        patch_m_1 = configuration["patch_m_1"]
        patch_c_1 = configuration["patch_c_1"]
        fc_k_1 = configuration["fc_k_1"]
        fc_w_1 = configuration["fc_w_1"]

        patch_n_2 = configuration["patch_n_2"]
        patch_c_2 = configuration["patch_c_2"]
        fc_k_2 = configuration["fc_k_2"]
        fc_w_2 = configuration["fc_w_2"]

        #self.validate(configuration)
            
        preprocessor = Transduction(self.device, self.input_shape)
        perceptron_flow = PerceptronMixer(
            self.device,
            self.input_shape, self.num_classes,
            patch_n_1, patch_m_1, patch_c_1, fc_w_1, fc_k_1,
            patch_n_2, patch_c_2, fc_w_2, fc_k_2)
        
        supermodel = Supermodel(self.device, self.input_shape, self.num_classes, preprocessor, perceptron_flow, self.pipeline_config["target_tq"])
        
        adaptation_manager = AdaptationManager()
        for perceptron in supermodel.get_perceptrons():
            perceptron.base_activation = LeakyGradReLU()
            #perceptron.base_activation = nn.LeakyReLU()
            adaptation_manager.update_activation(self.pipeline_config, perceptron)

            assert perceptron.layer.weight.shape[0] <= self.max_neurons, f"not enough neurons ({perceptron.layer.weight.shape[0]} > {self.max_neurons})"
            assert perceptron.layer.weight.shape[1] <= self.max_axons - 1, f"not enough axons ({perceptron.layer.weight.shape[1]} > {self.max_axons})"

        return supermodel