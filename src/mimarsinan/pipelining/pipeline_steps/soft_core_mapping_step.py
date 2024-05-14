from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.pipelining.pipeline_steps.perceptron_fusion_step import FusedLinear
from mimarsinan.mapping.mapping_utils import SoftCoreMapping
from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.models.layers import SavedTensorDecorator
from mimarsinan.models.layers import TransformedActivation

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory


import torch.nn as nn
import torch

import numpy as np

class SoftCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["soft_core_mapping"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry('model')

        for perceptron in model.get_perceptrons():
            if isinstance(perceptron.layer, FusedLinear):
                perceptron.layer = self.bring_back_bias(perceptron.layer)
        
        validator = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)

        self._calculate_input_activation_scales(model, validator, 1.0)

        bits = self.pipeline.config['weight_bits']
        q_max = (2 ** (bits - 1)) - 1
        soft_core_mapping = SoftCoreMapping(q_max = q_max)
        soft_core_mapping.map(model.get_mapper_repr())
        ChipLatency(soft_core_mapping).calculate()

        for core in soft_core_mapping.cores:

            scale = core.parameter_scale.cpu().numpy()
            assert np.allclose(
                core.core_matrix * scale, np.round(core.core_matrix * scale),
                atol=1e-3, rtol=1e-3), f"{core.core_matrix * scale}"

        self.add_entry("soft_core_mapping", soft_core_mapping, 'pickle')
    
    def _calculate_input_activation_scales(self, model, validator, rate):
        for perceptron in model.get_perceptrons():
            if not isinstance(perceptron.input_activation, TransformedActivation):
                perceptron.input_activation = TransformedActivation(perceptron.input_activation, [])
                
            perceptron.input_activation.decorate(SavedTensorDecorator())

        validator.validate()

        max_target_scale = 0.0
        for perceptron in model.get_perceptrons():
            saved_tensor_dec = perceptron.input_activation.pop_decorator()
            in_min = saved_tensor_dec.latest_input.min()
            in_max = saved_tensor_dec.latest_input.max()
            x = saved_tensor_dec.latest_input
            
            bins = 1000
            activation_hist = torch.histc(x.flatten(), bins=bins, min=in_min.item(), max=in_max.item())
            bin_edges = torch.linspace(in_min.item(), in_max.item(), steps=bins+1).to(self.pipeline.config['device'])

            activation_hist *= bin_edges[1:].to(self.pipeline.config['device'])
            activation_hist[activation_hist < 0] = 0
            hist_sum = activation_hist.sum()
            cumulative_hist = activation_hist.cumsum(0)
            cumulative_hist /= hist_sum

            clip_rate = 0.999
            
            # # find the index of the bin which first exceeds the rate
            index = (cumulative_hist > clip_rate).flatten().nonzero()[0].to(self.pipeline.config['device'])
            clipped_act_scale = bin_edges[index].item()

            target_act_scale = (in_max * (1.0 - rate) + rate * clipped_act_scale) 

            perceptron.set_input_activation_scale(target_act_scale)
            max_target_scale = max(max_target_scale, target_act_scale)

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