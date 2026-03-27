from mimarsinan.models.perceptron_mixer.simple_mlp import SimpleMLP
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ
from mimarsinan.pipelining.model_registry import ModelRegistry

import torch.nn as nn


@ModelRegistry.register("simple_mlp", label="Simple MLP", category="native")
class SimpleMLPBuilder:
    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        base_activation = configuration.get("base_activation", "ReLU")
        preprocessor = InputCQ(self.pipeline_config["target_tq"])

        perceptron_flow = SimpleMLP(self.device, self.input_shape, self.num_classes, configuration['mlp_width_1'], configuration['mlp_width_2'], base_activation_name=base_activation)
        supermodel = Supermodel(self.device, self.input_shape, self.num_classes, preprocessor, perceptron_flow, self.pipeline_config["target_tq"])
        return supermodel

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU", "LeakyReLU", "GELU"], "default": "ReLU"},
            {"key": "mlp_width_1", "type": "number", "label": "Hidden Width 1", "default": 256},
            {"key": "mlp_width_2", "type": "number", "label": "Hidden Width 2", "default": 128},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "mlp_width_1": [64, 128, 256, 512],
            "mlp_width_2": [64, 128, 256, 512],
        }