"""Builder that produces a shape-adapted ``torchvision`` ResNet-50."""

import torchvision.models as models

from mimarsinan.models.builders.torch.torchvision_builder_utils import (
    adapt_conv_in_channels,
    parse_image_input_shape,
    torchvision_workload_profile,
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

    workload_profile = staticmethod(torchvision_workload_profile)

    def get_pretrained_factory(self):
        """Return a callable that creates a pretrained ResNet-50 (ImageNet weights)."""

        def _factory():
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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
