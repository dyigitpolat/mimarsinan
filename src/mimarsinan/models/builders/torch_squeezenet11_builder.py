"""Builder that produces a standard ``torchvision`` SqueezeNet 1.1."""

import torchvision.models as models


class TorchSqueezeNet11Builder:
    def __init__(
        self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        return models.squeezenet1_1(num_classes=self.num_classes)

    def get_pretrained_factory(self):
        """Return a callable that creates a pretrained SqueezeNet 1.1 (ImageNet weights)."""
        def _factory():
            return models.squeezenet1_1(
                weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1
            )
        return _factory
