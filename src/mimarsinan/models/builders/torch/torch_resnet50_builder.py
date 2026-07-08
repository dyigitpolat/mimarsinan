"""Builder that produces a shape-adapted ``torchvision`` ResNet-50."""

import torchvision.models as models

from mimarsinan.common.workload_profile import ModelWorkloadProfile
from mimarsinan.models.builders.torch.torchvision_builder_utils import (
    adapt_conv_in_channels,
    parse_image_input_shape,
    torchvision_weight_set,
    torchvision_weights,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


def _adapt_resnet_for_input(model, input_shape):
    c, _, _ = parse_image_input_shape(input_shape, model_name="ResNet-50")
    model.conv1 = adapt_conv_in_channels(model.conv1, c)
    return model


@ModelRegistry.register("torch_resnet50", label="Torch ResNet-50", category="torch")
class TorchResNet50Builder:
    """Bottleneck-block residual net: conv/bn/relu/pool/linear + residual add, all groups==1."""

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        model = models.resnet50(num_classes=self.num_classes)
        return _adapt_resnet_for_input(model, self.input_shape)

    @classmethod
    def workload_profile(cls) -> ModelWorkloadProfile:
        """The pretrained weight sets torchvision ships for this backbone."""
        return ModelWorkloadProfile(pretrained_weight_sets=tuple(
            torchvision_weight_set(models.ResNet50_Weights[name])
            for name in ("IMAGENET1K_V1", "IMAGENET1K_V2")
        ))

    def get_pretrained_factory(self, weight_set_id: str | None = None):
        """Return a callable that creates a pretrained ResNet-50 for a registered weight set."""
        weights = torchvision_weights(type(self), models.ResNet50_Weights, weight_set_id)

        def _factory():
            model = models.resnet50(weights=weights)
            return _adapt_resnet_for_input(model, self.input_shape)

        return _factory

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU"], "default": "ReLU"},
        ]

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        """Accept any positive ``(C, H, W)`` and adapt the stem as needed."""
        try:
            parse_image_input_shape(input_shape, model_name="ResNet-50")
        except (TypeError, ValueError):
            return False
        return True
