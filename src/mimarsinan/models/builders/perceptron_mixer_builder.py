from mimarsinan.models.perceptron_mixer.perceptron_mixer import PerceptronMixer
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ
from mimarsinan.pipelining.model_registry import ModelRegistry

import torch.nn as nn


@ModelRegistry.register("mlp_mixer", label="MLP Mixer", category="native")
class PerceptronMixerBuilder:
    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        patch_n_1 = configuration["patch_n_1"]
        patch_m_1 = configuration["patch_m_1"]
        patch_c_1 = configuration["patch_c_1"]
        fc_w_1 = configuration["fc_w_1"]
        fc_w_2 = configuration["fc_w_2"]
            
        preprocessor = InputCQ(self.pipeline_config["target_tq"])
        perceptron_flow = PerceptronMixer(
            self.device,
            self.input_shape, self.num_classes,
            patch_n_1, patch_m_1, patch_c_1, fc_w_1, fc_w_2)
        
        supermodel = Supermodel(self.device, self.input_shape, self.num_classes, preprocessor, perceptron_flow, self.pipeline_config["target_tq"])
        
        allow_axon_tiling = bool(self.pipeline_config.get("allow_axon_tiling", False))
        for perceptron in supermodel.get_perceptrons():
            if not allow_axon_tiling:
                assert perceptron.layer.weight.shape[1] <= self.max_axons - 1, \
                    f"not enough axons ({perceptron.layer.weight.shape[1]} > {self.max_axons - 1})"

        return supermodel

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["LeakyReLU", "ReLU", "GELU"], "default": "LeakyReLU"},
            {"key": "patch_n_1", "type": "number", "label": "Patch Rows", "default": 4},
            {"key": "patch_m_1", "type": "number", "label": "Patch Cols", "default": 4},
            {"key": "patch_c_1", "type": "number", "label": "Patch Channels", "default": 32},
            {"key": "fc_w_1", "type": "number", "label": "FC Width 1", "default": 64},
            {"key": "fc_w_2", "type": "number", "label": "FC Width 2", "default": 64},
        ]