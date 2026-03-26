"""Builder that produces a shape-adapted ``torchvision`` SqueezeNet 1.1."""

import torchvision.models as models

from mimarsinan.models.builders.torchvision_builder_utils import (
    adapt_conv_in_channels,
    parse_image_input_shape,
)
from mimarsinan.pipelining.model_registry import ModelRegistry

def _adapt_squeezenet_for_input(model, input_shape):
    c, _, _ = parse_image_input_shape(input_shape, model_name="SqueezeNet")
    model.features[0] = adapt_conv_in_channels(model.features[0], c)
    return model


@ModelRegistry.register("torch_squeezenet11", label="Torch SqueezeNet", category="torch")
class TorchSqueezeNet11Builder:
    def __init__(
        self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        model = models.squeezenet1_1(num_classes=self.num_classes)
        return _adapt_squeezenet_for_input(model, self.input_shape)

    def get_pretrained_factory(self):
        """Return a callable that creates a pretrained SqueezeNet 1.1 (ImageNet weights)."""

        def _factory():
            model = models.squeezenet1_1(
                weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1
            )
            return _adapt_squeezenet_for_input(model, self.input_shape)

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
            parse_image_input_shape(input_shape, model_name="SqueezeNet")
        except (TypeError, ValueError):
            return False
        return True
