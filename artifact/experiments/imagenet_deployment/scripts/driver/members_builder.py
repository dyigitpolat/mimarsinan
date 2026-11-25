""

from __future__ import annotations

from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

import w38common as c

BUILDER_ID = "w38_torchvision"


def build_member(arch: str, input_shape, num_classes: int):
    from torchvision import models

    row = c.members().get(arch)
    if row is None:
        raise SystemExit(f"REFUSED: {arch!r} is no W38 member")
    crop = row["transforms"]["crop_size"][0]
    if tuple(int(v) for v in input_shape) != (3, crop, crop):
        raise SystemExit(f"REFUSED: {arch} input {tuple(input_shape)} is not its weights' crop (3, {crop}, {crop})")
    if int(num_classes) != 1000:
        raise SystemExit(f"REFUSED: {arch} keeps its 1,000-class head, not {num_classes}")
    return models.get_model(arch, weights=None).eval()


@ModelRegistry.register(BUILDER_ID, label="W38 torchvision ImageNet member", category="torch")
class W38TorchvisionBuilder:
    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device, self.input_shape, self.num_classes = device, input_shape, num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        return build_member(str((configuration or {}).get("arch")), self.input_shape, self.num_classes)

    @classmethod
    def get_config_schema(cls):
        return [{"key": "arch", "type": "select", "label": "Member", "default": "resnet50",
                 "options": sorted(c.members())}]

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        return str((config or {}).get("arch")) in c.members()
