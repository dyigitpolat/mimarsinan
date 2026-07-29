"""Builder for the CIFAR ResNet vehicle; registered as cifar_resnet20 (category torch)."""

from __future__ import annotations

from mimarsinan.models.cifar_models import CifarResNet
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register(
    "cifar_resnet20", label="CIFAR ResNet-20", category="torch"
)
class CifarResNet20Builder:
    """Builds the native ``CifarResNet``; TorchMappingStep converts it.

    Defaults are the standard ResNet-20 geometry (3 BasicBlocks per stage,
    16/32/64 channels); ``blocks_per_stage`` generalizes to the whole
    6n+2 CIFAR ResNet family (5 -> ResNet-32, 9 -> ResNet-56, ...).
    """

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration) -> CifarResNet:
        # A layout probe can build before the config form renders, so configuration may lack architecture keys.
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        return CifarResNet(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            blocks_per_stage=int(cfg["blocks_per_stage"]),
            base_width=int(cfg["base_width"]),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "blocks_per_stage", "type": "number", "label": "Blocks per Stage (n; depth 6n+2)", "default": 3, "min": 1, "max": 18},
            {"key": "base_width", "type": "number", "label": "Base Channels", "default": 16},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "blocks_per_stage": [2, 3, 5, 7, 9],
            "base_width": [8, 16, 24, 32],
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
        blocks = int(config.get("blocks_per_stage", 3))
        width = int(config.get("base_width", 16))
        return (
            blocks >= 1
            and width >= 1
            and h % 4 == 0 and w % 4 == 0  # two stride-2 stages
            and h >= 8 and w >= 8
        )
