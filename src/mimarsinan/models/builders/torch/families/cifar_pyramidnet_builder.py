"""Builder for the CIFAR additive PyramidNet; registered as cifar_pyramidnet (category torch)."""

from __future__ import annotations

from mimarsinan.models.families.pyramidnet import CifarPyramidNet, pyramidnet_units_per_group
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("cifar_pyramidnet", label="CIFAR PyramidNet", category="torch")
class CifarPyramidNetBuilder:
    """Builds ``CifarPyramidNet`` with basic units (defaults: depth 110, alpha 48)."""

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration) -> CifarPyramidNet:
        # A layout probe can build before the config form renders, so configuration may lack architecture keys.
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        return CifarPyramidNet(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            depth=int(cfg["depth"]),
            alpha=int(cfg["alpha"]),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "depth", "type": "number", "label": "Depth (6N + 2)", "default": 110, "min": 8},
            {"key": "alpha", "type": "number", "label": "Widening Factor alpha", "default": 48, "min": 0},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "depth": [38, 74, 110],
            "alpha": [48, 84, 150],
        }

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        shape = tuple(input_shape) if input_shape is not None else ()
        if len(shape) != 3:
            return False
        try:
            h, w = int(shape[1]), int(shape[2])
            pyramidnet_units_per_group(int(config.get("depth", 110)))
        except (TypeError, ValueError):
            return False
        return (
            int(config.get("alpha", 48)) >= 0
            and h % 4 == 0 and w % 4 == 0  # two stride-2 groups
            and h >= 8 and w >= 8
        )
