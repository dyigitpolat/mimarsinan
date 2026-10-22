"""Builder for CIFAR DenseNet-BC; registered as cifar_densenet_bc (category torch)."""

from __future__ import annotations

from mimarsinan.models.families.densenet_bc import (
    DENSENET_LAYER_ORDERS,
    CifarDenseNetBC,
    densenet_bc_layers_per_block,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("cifar_densenet_bc", label="CIFAR DenseNet-BC", category="torch")
class CifarDenseNetBCBuilder:
    """Builds ``CifarDenseNetBC`` (defaults: L = 100, k = 12, compression 0.5, published order)."""

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration) -> CifarDenseNetBC:
        # A layout probe can build before the config form renders, so configuration may lack architecture keys.
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        return CifarDenseNetBC(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            depth=int(cfg["depth"]),
            growth_rate=int(cfg["growth_rate"]),
            compression=float(cfg["compression"]),
            layer_order=str(cfg["layer_order"]),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "depth", "type": "number", "label": "Depth L (6n + 4)", "default": 100, "min": 10},
            {"key": "growth_rate", "type": "number", "label": "Growth Rate k", "default": 12, "min": 1},
            {"key": "compression", "type": "number", "label": "Compression theta", "default": 0.5},
            {"key": "layer_order", "type": "select", "label": "Layer Order",
             "options": list(DENSENET_LAYER_ORDERS), "default": "published"},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "depth": [40, 64, 100],
            "growth_rate": [12, 18, 24],
            "layer_order": list(DENSENET_LAYER_ORDERS),
        }

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        shape = tuple(input_shape) if input_shape is not None else ()
        if len(shape) != 3:
            return False
        try:
            h, w = int(shape[1]), int(shape[2])
            densenet_bc_layers_per_block(int(config.get("depth", 100)))
        except (TypeError, ValueError):
            return False
        compression = float(config.get("compression", 0.5))
        return (
            int(config.get("growth_rate", 12)) >= 1
            and 0.0 < compression <= 1.0
            and str(config.get("layer_order", "published")) in DENSENET_LAYER_ORDERS
            and h % 4 == 0 and w % 4 == 0  # two 2x2 average-pool transitions
            and h >= 8 and w >= 8
        )
