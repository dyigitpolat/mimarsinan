"""Builder for native PyTorch MLP-Mixer; registered as mlp_mixer with category torch."""

from mimarsinan.models.torch_mlp_mixer import TorchMLPMixer
from mimarsinan.pipelining.model_registry import ModelRegistry


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


@ModelRegistry.register("mlp_mixer", label="MLP Mixer", category="torch")
class TorchMLPMixerBuilder:
    """Builds native nn.Module MLP-Mixer; TorchMappingStep converts to Supermodel."""

    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        return TorchMLPMixer(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            patch_n_1=int(configuration["patch_n_1"]),
            patch_m_1=int(configuration["patch_m_1"]),
            patch_c_1=int(configuration["patch_c_1"]),
            fc_w_1=int(configuration["fc_w_1"]),
            fc_w_2=int(configuration["fc_w_2"]),
            base_activation=configuration.get("base_activation", "ReLU"),
        )

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
        h = int(input_shape[-2]) if input_shape is not None else 28
        w = int(input_shape[-1]) if input_shape is not None else 28
        return {
            "patch_n_1": _divisors(h),
            "patch_m_1": _divisors(w),
            "patch_c_1": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            "fc_w_1": [16, 32, 48, 64, 96, 128, 192, 256],
            "fc_w_2": [16, 32, 48, 64, 96, 128, 192, 256],
        }

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        pr = int(config.get("patch_n_1", 1))
        pc = int(config.get("patch_m_1", 1))
        h, w = int(input_shape[-2]), int(input_shape[-1])
        return h % pr == 0 and w % pc == 0
