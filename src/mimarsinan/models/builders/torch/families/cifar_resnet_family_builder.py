"""Builder for the CIFAR ResNet family (E294); registered as cifar_resnet_family (category torch)."""

from __future__ import annotations

from mimarsinan.models.families.cifar_resnet import (
    BLOCK_KINDS,
    BLOCK_ORDERS,
    MAX_BLOCKS_PER_STAGE,
    SHORTCUTS,
    CifarResNetFamily,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("cifar_resnet_family", label="CIFAR ResNet Family", category="torch")
class CifarResNetFamilyBuilder:
    """Builds ``CifarResNetFamily``; the defaults (and every ``cifar_resnet20`` config) give ``cifar_resnet20``.

    Members by configuration: ResNet-(6n+2) post-activation basic (n = 18: ResNet-110),
    WRN-(6n+4)-k (``width_multiplier`` k; pre- or post-activation) and the pre-activation
    bottleneck ResNet-(9n+2) (n = 18: ResNet-164, n = 111: ResNet-1001).
    """

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration) -> CifarResNetFamily:
        # A layout probe can build before the config form renders, so configuration may lack architecture keys.
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        stem_width = cfg["stem_width"]
        return CifarResNetFamily(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            blocks_per_stage=int(cfg["blocks_per_stage"]),
            base_width=int(cfg["base_width"]),
            stem_width=None if stem_width is None else int(stem_width),
            width_multiplier=int(cfg["width_multiplier"]),
            block_kind=str(cfg["block_kind"]),
            block_order=str(cfg["block_order"]),
            shortcut=str(cfg["shortcut"]),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "blocks_per_stage", "type": "number",
             "label": "Blocks per Stage (n; depth 6n + 2 basic, 9n + 2 bottleneck)",
             "default": 3, "min": 1, "max": MAX_BLOCKS_PER_STAGE},
            {"key": "base_width", "type": "number", "label": "Base Channels", "default": 16, "min": 1},
            {"key": "stem_width", "type": "number", "label": "Stem Channels (unset: Base Channels)",
             "default": None, "min": 1},
            {"key": "width_multiplier", "type": "number", "label": "Width Multiplier k", "default": 1, "min": 1},
            {"key": "block_kind", "type": "select", "label": "Block Kind",
             "options": list(BLOCK_KINDS), "default": "basic"},
            {"key": "block_order", "type": "select", "label": "Block Order",
             "options": list(BLOCK_ORDERS), "default": "post_activation"},
            {"key": "shortcut", "type": "select", "label": "Shortcut at a Shape Change",
             "options": list(SHORTCUTS), "default": "projection"},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "blocks_per_stage": [2, 3, 5, 9, 18],
            "base_width": [8, 16, 32],
            "width_multiplier": [1, 2, 4],
        }

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        shape = tuple(input_shape) if input_shape is not None else ()
        if len(shape) != 3:
            return False
        try:
            h, w = int(shape[1]), int(shape[2])
            blocks = int(config.get("blocks_per_stage", 3))
            widths = [int(config.get(k, d)) for k, d in (("base_width", 16), ("width_multiplier", 1))]
            stem = config.get("stem_width")
        except (TypeError, ValueError):
            return False
        return (
            1 <= blocks <= MAX_BLOCKS_PER_STAGE
            and min(widths) >= 1
            and (stem is None or int(stem) >= 1)
            and str(config.get("block_kind", "basic")) in BLOCK_KINDS
            and str(config.get("block_order", "post_activation")) in BLOCK_ORDERS
            and str(config.get("shortcut", "projection")) in SHORTCUTS
            and h % 4 == 0 and w % 4 == 0  # two stride-2 stages
            and h >= 8 and w >= 8
        )
