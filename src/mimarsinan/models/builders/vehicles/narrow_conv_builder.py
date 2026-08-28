"""Builder for NarrowConvNet; registered as narrow_conv (the fan-in bounded conv vehicle)."""

from mimarsinan.models.vehicles.narrow_conv import (
    ACTIVATED_READOUT,
    BARE_READOUT,
    NarrowConvNet,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("narrow_conv", label="Narrow Fan-In CNN", category="torch")
class NarrowConvBuilder:
    """Builds the native NarrowConvNet nn.Module; TorchMappingStep converts it."""

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        return NarrowConvNet(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            stem_channels=int(cfg["stem_channels"]),
            stem_kernel=int(cfg["stem_kernel"]),
            stem_stride=int(cfg["stem_stride"]),
            body_blocks=int(cfg["body_blocks"]),
            body_channels=int(cfg["body_channels"]),
            trunk_width=int(cfg["trunk_width"]),
            trunk_blocks=int(cfg["trunk_blocks"]),
            readout=str(cfg["readout"]),
            base_activation=cfg.get("base_activation", "ReLU"),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation",
             "options": ["ReLU", "LeakyReLU"], "default": "ReLU"},
            {"key": "stem_channels", "type": "number", "label": "Stem Channels",
             "default": 16, "min": 1},
            {"key": "stem_kernel", "type": "number", "label": "Stem Kernel",
             "default": 5, "min": 1},
            {"key": "stem_stride", "type": "number", "label": "Stem Stride",
             "default": 2, "min": 1},
            {"key": "body_blocks", "type": "number", "label": "Body Stages",
             "default": 2, "min": 0, "max": 5},
            {"key": "body_channels", "type": "number", "label": "Body Channels",
             "default": 16, "min": 1},
            {"key": "trunk_width", "type": "number", "label": "Trunk Width",
             "default": 128, "min": 1},
            {"key": "trunk_blocks", "type": "number", "label": "Trunk Stages",
             "default": 1, "min": 1, "max": 4},
            {"key": "readout", "type": "select", "label": "Readout",
             "options": [BARE_READOUT, ACTIVATED_READOUT], "default": BARE_READOUT},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "stem_channels": [8, 14, 16, 24],
            "stem_stride": [2, 4],
            "body_blocks": [0, 1, 2],
            "body_channels": [7, 14, 16],
            "trunk_width": [64, 120, 128],
            "trunk_blocks": [1, 2],
        }
