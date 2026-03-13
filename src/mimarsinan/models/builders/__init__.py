from .simple_mlp_builder import SimpleMLPBuilder
from .torch_mlp_mixer_builder import TorchMLPMixerBuilder
from .torch_vgg16_builder import TorchVGG16Builder
from .torch_vit_builder import TorchViTBuilder
from .torch_squeezenet11_builder import TorchSqueezeNet11Builder
from .torch_custom_builder import TorchCustomBuilder
from .torch_sequential_linear_builder import TorchSequentialLinearBuilder
from .torch_sequential_conv_builder import TorchSequentialConvBuilder

# Canonical registry: model_type id -> builder class. Used by pipeline steps and wizard schema.
BUILDERS_REGISTRY = {
    "mlp_mixer": TorchMLPMixerBuilder,
    "simple_mlp": SimpleMLPBuilder,
    "torch_vgg16": TorchVGG16Builder,
    "torch_vit": TorchViTBuilder,
    "torch_squeezenet11": TorchSqueezeNet11Builder,
    "torch_custom": TorchCustomBuilder,
    "torch_sequential_linear": TorchSequentialLinearBuilder,
    "torch_sequential_conv": TorchSequentialConvBuilder,
}