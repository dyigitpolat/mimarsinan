"""Builder for the MHA-leaf vision transformer; registered as torch_vit_leaf (category torch)."""

from __future__ import annotations

from mimarsinan.models.vit_leaf import LeafVisionTransformer
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


def _parse_square_image_shape(input_shape) -> tuple[int, int]:
    """(C, H, W) with H == W -> (channels, size)."""
    if len(input_shape) != 3:
        raise ValueError(
            f"MHA-leaf ViT expects a (C, H, W) input shape; got {tuple(input_shape)}"
        )
    c, h, w = (int(d) for d in input_shape)
    if h != w:
        raise ValueError(f"MHA-leaf ViT expects square images; got {h}x{w}")
    return c, h


@ModelRegistry.register(
    "torch_vit_leaf", label="Torch ViT (MHA leaf)", category="torch"
)
class TorchViTLeafBuilder:
    """Builds a native ``LeafVisionTransformer``; TorchMappingStep converts it.

    Defaults are the DeiT-Tiny geometry (d=192, 3 heads, depth 12, patch 16 —
    197 tokens at 224px).
    """

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration) -> LeafVisionTransformer:
        channels, size = _parse_square_image_shape(self.input_shape)
        return LeafVisionTransformer(
            image_size=size,
            patch_size=int(configuration.get("patch_size", 16)),
            in_channels=channels,
            embed_dim=int(configuration.get("embed_dim", 192)),
            num_heads=int(configuration.get("num_heads", 3)),
            depth=int(configuration.get("depth", 12)),
            mlp_ratio=float(configuration.get("mlp_ratio", 4.0)),
            num_classes=self.num_classes,
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "patch_size", "type": "number", "label": "Patch Size", "default": 16},
            {"key": "embed_dim", "type": "number", "label": "Embedding Dim", "default": 192},
            {"key": "num_heads", "type": "number", "label": "Attention Heads", "default": 3},
            {"key": "depth", "type": "number", "label": "Encoder Blocks", "default": 12},
            {"key": "mlp_ratio", "type": "number", "label": "MLP Ratio", "default": 4.0},
        ]

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        try:
            _, size = _parse_square_image_shape(input_shape)
        except (TypeError, ValueError):
            return False
        patch_size = int(config.get("patch_size", 16))
        embed_dim = int(config.get("embed_dim", 192))
        num_heads = int(config.get("num_heads", 3))
        depth = int(config.get("depth", 12))
        return (
            patch_size > 0
            and size % patch_size == 0
            and num_heads > 0
            and embed_dim % num_heads == 0
            and depth > 0
        )
