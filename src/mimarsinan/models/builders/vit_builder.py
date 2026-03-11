"""
Builder for the Vision Transformer (ViT) model.

Accepts a configuration dictionary (from user specification or
architecture search) and produces a Supermodel wrapping a
VisionTransformer PerceptronFlow.
"""

from mimarsinan.models.perceptron_mixer.vision_transformer import VisionTransformer
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ
from mimarsinan.pipelining.model_registry import ModelRegistry


@ModelRegistry.register("vit", label="ViT", category="native")
class VitBuilder:
    """
    Build a Supermodel containing a VisionTransformer.

    Expected configuration keys:
        patch_size   : int   (e.g. 4)
        d_model      : int   (e.g. 128)
        num_heads    : int   (e.g. 4)
        num_layers   : int   (e.g. 4)
        mlp_ratio    : int   (e.g. 4)
        dropout      : float (e.g. 0.1)   [optional, default 0.1]
    """

    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        patch_size = int(configuration["patch_size"])
        d_model = int(configuration["d_model"])
        num_heads = int(configuration["num_heads"])
        num_layers = int(configuration["num_layers"])
        mlp_ratio = int(configuration["mlp_ratio"])
        dropout = float(configuration.get("dropout", 0.1))
        base_activation = configuration.get("base_activation", "ReLU")

        preprocessor = InputCQ(self.pipeline_config["target_tq"])
        perceptron_flow = VisionTransformer(
            device=self.device,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            patch_size=patch_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            base_activation_name=base_activation,
        )

        supermodel = Supermodel(
            self.device,
            self.input_shape,
            self.num_classes,
            preprocessor,
            perceptron_flow,
            self.pipeline_config["target_tq"],
        )

        # Validate axon constraints for Perceptrons
        allow_axon_tiling = bool(self.pipeline_config.get("allow_axon_tiling", False))
        if not allow_axon_tiling:
            for perceptron in supermodel.get_perceptrons():
                n_axons = perceptron.layer.weight.shape[1]
                assert n_axons <= self.max_axons - 1, (
                    f"ViT perceptron '{getattr(perceptron, 'name', '?')}' "
                    f"needs {n_axons} axons but max_axons={self.max_axons} "
                    f"(bias takes 1). Enable allow_axon_tiling or increase max_axons."
                )

        return supermodel

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU", "LeakyReLU", "GELU"], "default": "ReLU"},
            {"key": "patch_size", "type": "number", "label": "Patch Size", "default": 4},
            {"key": "d_model", "type": "number", "label": "Embed Dim", "default": 128},
            {"key": "num_heads", "type": "number", "label": "Num Heads", "default": 4},
            {"key": "num_layers", "type": "number", "label": "Num Layers", "default": 4},
            {"key": "mlp_ratio", "type": "number", "label": "MLP Ratio", "default": 4},
            {"key": "dropout", "type": "number", "label": "Dropout", "default": 0.1, "step": 0.05},
        ]

    @classmethod
    def get_nas_search_options(cls, input_shape=None):
        """Discrete search values for numeric config keys used by the NAS."""
        return {
            "patch_size": [2, 4, 8],
            "d_model":    [64, 96, 128, 192, 256],
            "num_heads":  [2, 4, 8],
            "num_layers": [2, 4, 6],
            "mlp_ratio":  [2, 4],
        }

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape, allow_axon_tiling):
        """patch_size must divide image dims; d_model must be divisible by num_heads."""
        patch_size = int(config.get("patch_size", 4))
        d_model = int(config.get("d_model", 128))
        num_heads = int(config.get("num_heads", 4))
        H, W = int(input_shape[-2]), int(input_shape[-1])
        if H % patch_size != 0 or W % patch_size != 0:
            return False
        if d_model % num_heads != 0:
            return False
        return True
