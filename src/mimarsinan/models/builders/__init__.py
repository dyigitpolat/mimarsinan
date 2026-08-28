from .simple_mlp_builder import SimpleMLPBuilder as SimpleMLPBuilder
from .torch_mlp_mixer_builder import TorchMLPMixerBuilder as TorchMLPMixerBuilder
from .torch_mlp_mixer_core_builder import TorchMLPMixerCoreBuilder as TorchMLPMixerCoreBuilder
from .torch.torch_vgg16_builder import TorchVGG16Builder as TorchVGG16Builder
from .torch.torch_vit_builder import TorchViTBuilder as TorchViTBuilder
from .torch.torch_squeezenet11_builder import TorchSqueezeNet11Builder as TorchSqueezeNet11Builder
from .torch.torch_resnet50_builder import TorchResNet50Builder as TorchResNet50Builder
from .torch.torch_vit_leaf_builder import TorchViTLeafBuilder as TorchViTLeafBuilder
from .torch.cifar_resnet20_builder import CifarResNet20Builder as CifarResNet20Builder
from .torch.cifar_vgg8_builder import CifarVGG8Builder as CifarVGG8Builder
from .torch.cifar_vit_leaf_builder import CifarViTLeafBuilder as CifarViTLeafBuilder
from .torch_custom_builder import TorchCustomBuilder as TorchCustomBuilder
from .torch_sequential_linear_builder import TorchSequentialLinearBuilder as TorchSequentialLinearBuilder
from .torch_sequential_conv_builder import TorchSequentialConvBuilder as TorchSequentialConvBuilder
from .deep_mlp_builder import DeepMLPBuilder as DeepMLPBuilder
from .deep_cnn_builder import DeepCNNBuilder as DeepCNNBuilder
from .vehicles.stream_cnn_builder import StreamCNNBuilder as StreamCNNBuilder
from .vehicles.narrow_conv_builder import NarrowConvBuilder as NarrowConvBuilder
from .lenet5_builder import LeNet5Builder as LeNet5Builder

from typing import Any, Callable, cast

from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.platform.packaging_contract import (
    SPIKING_PACKAGING,
    PackagingContract,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

# One builder SSOT: the imports above ran every @ModelRegistry.register, so this
# view IS the registry (no hand-maintained duplicate mapping).
BUILDERS_REGISTRY = ModelRegistry.builder_classes()


def build_model(
    builder: Any,
    model_config: Any,
    *,
    encoding_placement: str,
    packaging: PackagingContract = SPIKING_PACKAGING,
) -> Any:
    """Build ``builder``'s model with its encoding-layer placement resolved.

    ``encoding_layer_placement`` is a config decision, not a builder decision: a
    builder that bakes it makes the knob a silent no-op for its own model type,
    and only for that one type. So builders build, and the placement is resolved
    HERE, once, at the moment a mapper flow first exists.

    A ``native``-category builder returns the flow itself, so this is that
    moment. A ``torch``-category builder returns an ``nn.Module`` with no mapper
    graph yet; its flow is born later in ``convert_torch_model``, which applies
    the same placement through the same single writer (``mark_encoding_layers``).
    Nothing else marks. ``packaging`` is the deployment's packaging contract, so
    a value-domain build is stamped not-applicable by the same writer instead of
    leaving the graph unstamped.
    """
    model = builder.build(model_config)
    get_mapper_repr = getattr(model, "get_mapper_repr", None)
    if callable(get_mapper_repr):
        mapper_repr = cast(Callable[[], ModelRepresentation], get_mapper_repr)()
        mark_encoding_layers(
            mapper_repr, placement=encoding_placement, packaging=packaging,
        )
    return model
