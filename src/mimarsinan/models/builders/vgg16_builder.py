from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ
from mimarsinan.models.vgg16 import VGG16Mapper


class VGG16Builder:
    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        # VGG16 has a fixed architecture; configuration is currently unused but kept
        # for parity with other builders.
        _ = configuration

        preprocessor = InputCQ(self.pipeline_config["target_tq"])
        perceptron_flow = VGG16Mapper(
            self.device,
            self.input_shape,
            self.num_classes,
            max_axons=self.max_axons,
            max_neurons=self.max_neurons,
        )
        supermodel = Supermodel(
            self.device,
            self.input_shape,
            self.num_classes,
            preprocessor,
            perceptron_flow,
            self.pipeline_config["target_tq"],
        )

        # Enforce axon constraints only when axon-tiling is disabled.
        # (Neuron tiling is handled at mapping time by splitting output rows.)
        allow_axon_tiling = bool(self.pipeline_config.get("allow_axon_tiling", False))
        if not allow_axon_tiling:
            for perceptron in supermodel.get_perceptrons():
                in_axons = perceptron.layer.weight.shape[1]
                if in_axons > self.max_axons - 1:
                    raise ValueError(
                        f"not enough axons ({in_axons} > {self.max_axons - 1})"
                    )

        return supermodel


