"""Builder for the CIFAR leaf-ViT vehicle; registered as cifar_vit_leaf (category torch)."""

from __future__ import annotations

from mimarsinan.models.vit_leaf import LeafVisionTransformer
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.models.builders.torch.torch_vit_leaf_builder import (
    _parse_square_image_shape,
)


@ModelRegistry.register(
    "cifar_vit_leaf", label="CIFAR ViT (MHA leaf)", category="torch"
)
class CifarViTLeafBuilder:
    """Builds a native ``LeafVisionTransformer`` at the CIFAR named config.

    Same builder contract as ``torch_vit_leaf`` but with the ``cifar_vit_leaf``
    geometry as the defaults: 32px/patch 4 (65 tokens), d=192, 3 heads,
    depth 7, mlp_ratio 2.0 — the compact from-scratch ViT-CIFAR recipe scale.
    """

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration) -> LeafVisionTransformer:
        channels, size = _parse_square_image_shape(self.input_shape)
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        return LeafVisionTransformer(
            image_size=size,
            patch_size=int(cfg["patch_size"]),
            in_channels=channels,
            embed_dim=int(cfg["embed_dim"]),
            num_heads=int(cfg["num_heads"]),
            depth=int(cfg["depth"]),
            mlp_ratio=float(cfg["mlp_ratio"]),
            num_classes=self.num_classes,
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "patch_size", "type": "number", "label": "Patch Size", "default": 4},
            {"key": "embed_dim", "type": "number", "label": "Embedding Dim", "default": 192},
            {"key": "num_heads", "type": "number", "label": "Attention Heads", "default": 3},
            {"key": "depth", "type": "number", "label": "Encoder Blocks", "default": 7},
            {"key": "mlp_ratio", "type": "number", "label": "MLP Ratio", "default": 2.0},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "embed_dim": [96, 192, 384],
            "num_heads": [3, 6],
            "depth": [4, 7, 9],
        }

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        try:
            _, size = _parse_square_image_shape(input_shape)
        except (TypeError, ValueError):
            return False
        patch_size = int(config.get("patch_size", 4))
        embed_dim = int(config.get("embed_dim", 192))
        num_heads = int(config.get("num_heads", 3))
        depth = int(config.get("depth", 7))
        return (
            patch_size > 0
            and size % patch_size == 0
            and num_heads > 0
            and embed_dim % num_heads == 0
            and depth > 0
        )
