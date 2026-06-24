"""Builder for DeepCNN; registered as deep_cnn with category torch (deep-conv depth probe)."""

from mimarsinan.models.deep_cnn import DeepCNN
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("deep_cnn", label="Deep CNN", category="torch")
class DeepCNNBuilder:
    """Builds the native DeepCNN nn.Module; TorchMappingStep converts it to a flow."""

    def __init__(self, device, input_shape, num_classes, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        # A layout probe may fire before the model_config form has rendered, in
        # which case ``configuration`` arrives without the architecture keys.
        # Fall back to the schema defaults so the probe builds a valid model.
        schema_defaults = {f["key"]: f.get("default") for f in self.get_config_schema()}
        cfg = {**schema_defaults, **(configuration or {})}
        return DeepCNN(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            depth=int(cfg["depth"]),
            width=int(cfg["width"]),
            base_activation=cfg.get("base_activation", "ReLU"),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU", "LeakyReLU", "GELU"], "default": "ReLU"},
            {"key": "depth", "type": "number", "label": "Conv Blocks (depth)", "default": 8, "min": 4, "max": 16},
            {"key": "width", "type": "number", "label": "Base Channels (width)", "default": 16},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "depth": [4, 6, 8, 10, 12, 14, 16],
            "width": [8, 16, 24, 32],
        }
