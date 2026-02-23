"""
Builder for the Vision Transformer (ViT) model.

Accepts a configuration dictionary (from user specification or
architecture search) and produces a Supermodel wrapping a
VisionTransformer PerceptronFlow.
"""

from mimarsinan.models.perceptron_mixer.vision_transformer import VisionTransformer
from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ


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

