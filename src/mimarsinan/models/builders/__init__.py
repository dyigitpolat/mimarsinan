from .simple_mlp_builder import SimpleMLPBuilder
from .torch_mlp_mixer_builder import TorchMLPMixerBuilder
from .torch_mlp_mixer_core_builder import TorchMLPMixerCoreBuilder
from .torch.torch_vgg16_builder import TorchVGG16Builder
from .torch.torch_vit_builder import TorchViTBuilder
from .torch.torch_squeezenet11_builder import TorchSqueezeNet11Builder
from .torch.torch_resnet50_builder import TorchResNet50Builder
from .torch_custom_builder import TorchCustomBuilder
from .torch_sequential_linear_builder import TorchSequentialLinearBuilder
from .torch_sequential_conv_builder import TorchSequentialConvBuilder
from .deep_mlp_builder import DeepMLPBuilder
from .deep_cnn_builder import DeepCNNBuilder

from .lenet5_builder import LeNet5Builder

# Canonical registry: model_type id -> builder class. Used by pipeline steps and wizard schema.
BUILDERS_REGISTRY = {
    "mlp_mixer": TorchMLPMixerBuilder,
    "mlp_mixer_core": TorchMLPMixerCoreBuilder,
    "simple_mlp": SimpleMLPBuilder,
    "torch_vgg16": TorchVGG16Builder,
    "torch_vit": TorchViTBuilder,
    "torch_squeezenet11": TorchSqueezeNet11Builder,
    "torch_resnet50": TorchResNet50Builder,
    "torch_custom": TorchCustomBuilder,
    "torch_sequential_linear": TorchSequentialLinearBuilder,
    "torch_sequential_conv": TorchSequentialConvBuilder,
    "deep_mlp": DeepMLPBuilder,
    "deep_cnn": DeepCNNBuilder,

    "lenet5": LeNet5Builder,
}