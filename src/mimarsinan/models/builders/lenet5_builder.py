"""Builder for LeNet5; registered as lenet5 with category torch (T1 classical rung)."""

from mimarsinan.models.lenet5 import LeNet5
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


@ModelRegistry.register("lenet5", label="LeNet-5", category="torch")
class LeNet5Builder:
    """Builds the native LeNet5 nn.Module; TorchMappingStep converts it to a flow."""

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
        return LeNet5(
            input_shape=tuple(self.input_shape),
            num_classes=self.num_classes,
            base_activation=cfg.get("base_activation", "ReLU"),
        )

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "variant", "type": "select", "label": "Variant", "options": ["lenet5"], "default": "lenet5"},
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU", "LeakyReLU", "GELU"], "default": "ReLU"},
        ]
