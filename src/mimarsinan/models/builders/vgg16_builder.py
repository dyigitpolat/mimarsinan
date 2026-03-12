from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ
from mimarsinan.models.vgg16 import VGG16Mapper
from mimarsinan.pipelining.model_registry import ModelRegistry


@ModelRegistry.register("vgg16", label="VGG-16", category="native")
class VGG16Builder:
    def __init__(self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        base_activation = (configuration or {}).get("base_activation", "ReLU")

        preprocessor = InputCQ(self.pipeline_config["target_tq"])
        perceptron_flow = VGG16Mapper(
            self.device,
            self.input_shape,
            self.num_classes,
            max_axons=self.max_axons,
            max_neurons=self.max_neurons,
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

        return supermodel

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU", "LeakyReLU"], "default": "ReLU"},
        ]


