from .simple_mlp_builder import SimpleMLPBuilder as SimpleMLPBuilder
from .torch_mlp_mixer_builder import TorchMLPMixerBuilder as TorchMLPMixerBuilder
from .torch_mlp_mixer_core_builder import TorchMLPMixerCoreBuilder as TorchMLPMixerCoreBuilder
from .torch.torch_vgg16_builder import TorchVGG16Builder as TorchVGG16Builder
from .torch.torch_vit_builder import TorchViTBuilder as TorchViTBuilder
from .torch.torch_squeezenet11_builder import TorchSqueezeNet11Builder as TorchSqueezeNet11Builder
from .torch.torch_resnet50_builder import TorchResNet50Builder as TorchResNet50Builder
from .torch_custom_builder import TorchCustomBuilder as TorchCustomBuilder
from .torch_sequential_linear_builder import TorchSequentialLinearBuilder as TorchSequentialLinearBuilder
from .torch_sequential_conv_builder import TorchSequentialConvBuilder as TorchSequentialConvBuilder
from .deep_mlp_builder import DeepMLPBuilder as DeepMLPBuilder
from .deep_cnn_builder import DeepCNNBuilder as DeepCNNBuilder
from .lenet5_builder import LeNet5Builder as LeNet5Builder

from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

# One builder SSOT: the imports above ran every @ModelRegistry.register, so this
# view IS the registry (no hand-maintained duplicate mapping).
BUILDERS_REGISTRY = ModelRegistry.builder_classes()