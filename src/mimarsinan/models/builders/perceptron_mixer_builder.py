from mimarsinan.models.perceptron_mixer.perceptron_mixer import PerceptronMixer
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ
from mimarsinan.pipelining.model_registry import ModelRegistry

import torch.nn as nn


def _divisors(n):
    n = int(n)
    return [d for d in range(1, n + 1) if n % d == 0]


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
        base_activation = configuration.get("base_activation", "ReLU")

        preprocessor = InputCQ(self.pipeline_config["target_tq"])
        perceptron_flow = PerceptronMixer(
            self.device,
            self.input_shape, self.num_classes,
            patch_n_1, patch_m_1, patch_c_1, fc_w_1, fc_w_2,
            base_activation_name=base_activation)

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

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        """Discrete search values for numeric config keys used by the NAS."""
        h = int(input_shape[-2]) if input_shape is not None else 28
        w = int(input_shape[-1]) if input_shape is not None else 28
        return {
            "patch_n_1": _divisors(h),
            "patch_m_1": _divisors(w),
            "patch_c_1": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            "fc_w_1":    [16, 32, 48, 64, 96, 128, 192, 256],
            "fc_w_2":    [16, 32, 48, 64, 96, 128, 192, 256],
        }

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape, allow_axon_tiling):
        """Patch dimensions must divide the spatial input dimensions."""
        pr = int(config.get("patch_n_1", 1))
        pc = int(config.get("patch_m_1", 1))
        h, w = int(input_shape[-2]), int(input_shape[-1])
        return h % pr == 0 and w % pc == 0
