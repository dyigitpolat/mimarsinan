"""Builder for ViT-Lite with the CCT encoder block; registered as vit_lite_cct (category torch)."""

from __future__ import annotations

from mimarsinan.models.families.vit_lite import ATTENTION_FORMS, ViTLiteCCT
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("vit_lite_cct", label="ViT-Lite (CCT block)", category="torch")
class ViTLiteCCTBuilder:
    """Builds ``ViTLiteCCT`` (defaults: ViT-Lite-7/4, the ViT gate's network)."""

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration) -> ViTLiteCCT:
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        channels, size, _width = (int(d) for d in self.input_shape)
        return ViTLiteCCT(
            num_classes=self.num_classes, img_size=size, in_channels=channels, patch=int(cfg["patch_size"]),
            dim=int(cfg["embed_dim"]), depth=int(cfg["depth"]), heads=int(cfg["num_heads"]),
            mlp_ratio=float(cfg["mlp_ratio"]), attention=str(cfg["attention"]),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "patch_size", "type": "number", "label": "Patch Size", "default": 4},
            {"key": "embed_dim", "type": "number", "label": "Embedding Dim", "default": 256},
            {"key": "num_heads", "type": "number", "label": "Attention Heads", "default": 4},
            {"key": "depth", "type": "number", "label": "Encoder Blocks", "default": 7},
            {"key": "mlp_ratio", "type": "number", "label": "MLP Ratio", "default": 2.0},
            {"key": "attention", "type": "select", "label": "Attention Form", "default": "softmax",
             "options": list(ATTENTION_FORMS)},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {"embed_dim": [128, 256], "num_heads": [2, 4], "depth": [4, 7]}

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        shape = tuple(input_shape) if input_shape is not None else ()
        if len(shape) != 3 or int(shape[1]) != int(shape[2]):
            return False
        patch, dim, heads = int(config.get("patch_size", 4)), int(config.get("embed_dim", 256)), int(
            config.get("num_heads", 4))
        return patch > 0 and int(shape[1]) % patch == 0 and heads > 0 and dim % heads == 0 and \
            int(config.get("depth", 7)) > 0 and config.get("attention", "softmax") in ATTENTION_FORMS
