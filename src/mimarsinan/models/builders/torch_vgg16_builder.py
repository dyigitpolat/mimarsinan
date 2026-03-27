"""Builder that produces a shape-adapted ``torchvision`` VGG-16 with batch norm."""

import torch.nn as nn
import torchvision.models as models

from mimarsinan.pipelining.model_registry import ModelRegistry
from mimarsinan.models.builders.torchvision_builder_utils import (
    adapt_conv_in_channels,
    parse_image_input_shape,
)

_VGG_MAX_POOL_COUNT = 5


def _kept_vgg_pool_count(h: int, w: int) -> int:
    pooled = min(h, w)
    keep = 0
    while pooled >= 2 and keep < _VGG_MAX_POOL_COUNT:
        pooled //= 2
        keep += 1
    return keep


def _adapt_vgg_for_input(model: nn.Module, input_shape) -> nn.Module:
    c, h, w = parse_image_input_shape(input_shape, model_name="VGG16")
    model.features[0] = adapt_conv_in_channels(model.features[0], c)

    # Drop trailing pools when a small input would otherwise collapse below 1x1.
    max_pool_indices = [
        idx for idx, mod in enumerate(model.features) if isinstance(mod, nn.MaxPool2d)
    ]
    keep_pool_count = _kept_vgg_pool_count(h, w)
    for idx in max_pool_indices[keep_pool_count:]:
        model.features[idx] = nn.Identity()
    return model


@ModelRegistry.register("torch_vgg16", label="Torch VGG16", category="torch")
class TorchVGG16Builder:
    def __init__(
        self, device, input_shape, num_classes, pipeline_config
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        model = models.vgg16_bn(num_classes=self.num_classes)
        return _adapt_vgg_for_input(model, self.input_shape)

    def get_pretrained_factory(self):
        """Return a callable that creates a pretrained VGG16-BN (ImageNet weights)."""

        def _factory():
            model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            return _adapt_vgg_for_input(model, self.input_shape)

        return _factory

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU", "LeakyReLU"], "default": "ReLU"},
        ]

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        """Accept any positive ``(C, H, W)`` and adapt the stem/pooling as needed."""
        try:
            parse_image_input_shape(input_shape, model_name="VGG16")
        except (TypeError, ValueError):
            return False
        return True
