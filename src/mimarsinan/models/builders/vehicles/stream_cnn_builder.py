"""Builder for StreamCNN; registered as stream_cnn (the streamed-lif conv vehicle)."""

from mimarsinan.models.vehicles.stream_cnn import StreamCNN
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("stream_cnn", label="Stream CNN", category="torch")
class StreamCNNBuilder:
    """Builds the native StreamCNN nn.Module; TorchMappingStep converts it."""

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        return StreamCNN(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            width=int(cfg["width"]),
            blocks=int(cfg["blocks"]),
            fc_width=int(cfg["fc_width"]),
            base_activation=cfg.get("base_activation", "ReLU"),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation",
             "options": ["ReLU", "LeakyReLU"], "default": "ReLU"},
            {"key": "width", "type": "number", "label": "Base Channels", "default": 16},
            {"key": "blocks", "type": "number", "label": "Stride-2 Blocks",
             "default": 3, "min": 2, "max": 5},
            {"key": "fc_width", "type": "number", "label": "FC Width", "default": 128},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "width": [8, 16, 24, 32],
            "blocks": [2, 3, 4],
            "fc_width": [64, 128, 256],
        }
