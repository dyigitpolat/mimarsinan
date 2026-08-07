"""Builder for the CIFAR VGG-8 vehicle; registered as cifar_vgg8 (category torch)."""

from __future__ import annotations

from mimarsinan.models.cifar_models import CifarVGG8
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("cifar_vgg8", label="CIFAR VGG-8", category="torch")
class CifarVGG8Builder:
    """Builds the native ``CifarVGG8``; TorchMappingStep converts it.

    Defaults are the standard CIFAR VGG-8 scale ([64,64]-[128,128]-[256,256]
    trunk, 512-wide FC): worst fan-in 2304 (+1 softcore bias row = 2305), so
    a 2560-axon core hosts it without input splitting.
    """

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration) -> CifarVGG8:
        # A layout probe can build before the config form renders, so configuration may lack architecture keys.
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        return CifarVGG8(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            base_channels=int(cfg["base_channels"]),
            fc_width=int(cfg["fc_width"]),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_channels", "type": "number", "label": "Base Channels", "default": 64},
            {"key": "fc_width", "type": "number", "label": "FC Hidden Width", "default": 512},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "base_channels": [16, 32, 64],
            "fc_width": [128, 256, 512],
        }

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        shape = tuple(input_shape) if input_shape is not None else ()
        if len(shape) != 3:
            return False
        try:
            h, w = int(shape[1]), int(shape[2])
        except (TypeError, ValueError):
            return False
        base_channels = int(config.get("base_channels", 64))
        fc_width = int(config.get("fc_width", 512))
        return (
            base_channels >= 1
            and fc_width >= 1
            and h % 8 == 0 and w % 8 == 0  # three /2 maxpools
            and h >= 16 and w >= 16
        )
