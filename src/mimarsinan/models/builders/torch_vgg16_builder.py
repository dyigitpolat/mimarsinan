"""Builder that produces a standard ``torchvision`` VGG-16 with batch norm."""

import torchvision.models as models


class TorchVGG16Builder:
    def __init__(
        self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        return models.vgg16_bn(num_classes=self.num_classes)

    def get_pretrained_factory(self):
        """Return a callable that creates a pretrained VGG16-BN (ImageNet weights)."""
        def _factory():
            return models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        return _factory
