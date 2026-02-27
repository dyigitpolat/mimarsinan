"""Builder that produces a standard ``torchvision`` Vision Transformer."""

import torchvision.models as models


class TorchViTBuilder:
    def __init__(
        self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        image_size = self.input_shape[-1]
        return models.vit_b_16(
            image_size=image_size,
            num_classes=self.num_classes,
        )

    def get_pretrained_factory(self):
        """Return a callable that creates a pretrained ViT-B/16 (ImageNet weights)."""
        def _factory():
            return models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        return _factory
