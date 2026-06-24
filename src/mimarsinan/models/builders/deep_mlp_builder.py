"""Builder for DeepMLP; registered as deep_mlp with category torch."""

from mimarsinan.models.deep_mlp import DeepMLP
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("deep_mlp", label="Deep MLP", category="torch")
class DeepMLPBuilder:
    """Builds the native DeepMLP nn.Module; TorchMappingStep converts it to a flow."""

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
        return DeepMLP(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            depth=int(cfg["depth"]),
            width=int(cfg["width"]),
            base_activation=cfg.get("base_activation", "ReLU"),
            residual=bool(cfg.get("residual", False)),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU", "LeakyReLU", "GELU"], "default": "ReLU"},
            {"key": "depth", "type": "number", "label": "Hidden Layers (depth)", "default": 8, "min": 4, "max": 32},
            {"key": "width", "type": "number", "label": "Hidden Width", "default": 64},
            {"key": "residual", "type": "toggle", "label": "Residual Skips (equal-width)", "default": False},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        return {
            "depth": [4, 6, 8, 12, 16, 24, 32],
            "width": [32, 48, 64, 96, 128],
        }
