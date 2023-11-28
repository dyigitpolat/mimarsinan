from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.pipelining.pipeline_steps.perceptron_fusion_step import FusedLinear
from mimarsinan.mapping.mapping_utils import SoftCoreMapping
from mimarsinan.mapping.chip_latency import ChipLatency
import torch.nn as nn

class SoftCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["soft_core_mapping"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def bring_back_bias(self, fused_linear_layer):
        assert isinstance(fused_linear_layer, FusedLinear), 'Input layer must be an instance of LinearWithoutBias'
        
        # Get the weights from the existing layer
        weights = fused_linear_layer.linear.weight.data
        
        # Split the weights back into the main weights and the bias
        main_weights, bias = weights[:, :-1], weights[:, -1]

        # Create a new layer with the main weights and bias
        out_features, in_features = main_weights.shape
        new_layer = nn.Linear(in_features, out_features)
        new_layer.weight.data = main_weights
        new_layer.bias.data = bias

        return new_layer

    def process(self):
        model = self.get_entry('model')

        for perceptron in model.get_perceptrons():
            if isinstance(perceptron.layer, FusedLinear):
                perceptron.layer = self.bring_back_bias(perceptron.layer)

        soft_core_mapping = SoftCoreMapping()
        soft_core_mapping.map(model.get_mapper_repr())
        ChipLatency(soft_core_mapping).calculate()

        self.add_entry("soft_core_mapping", soft_core_mapping, 'pickle')