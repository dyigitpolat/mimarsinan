from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow

import torch.nn as nn
import einops

class SimpleMLP(PerceptronFlow):
    def __init__(
        self, 
        device,
        input_shape,
        num_classes,
        mlp_width, 
        extra_layers):
        super(SimpleMLP, self).__init__(device)

        self.input_activation = nn.Identity()
        
        self.input_shape = input_shape
        self.input_width = input_shape[-3] * input_shape[-2] * input_shape[-1]

        network_shape = [self.input_width] + ([mlp_width] * extra_layers) + [num_classes]

        self.perceptrons = nn.ModuleList()
        for i in range(len(network_shape) - 1):
            self.perceptrons.append(
                Perceptron(
                    output_channels=network_shape[i+1], 
                    input_features=network_shape[i], 
                    normalization=nn.LazyBatchNorm1d()
                )
            )

    def get_input_activation(self):
        return self.input_activation
    
    def set_input_activation(self, activation):
        self.input_activation = activation

    def get_perceptrons(self):
        return self.perceptrons
    
    def get_mapper_repr(self):
        out = InputMapper(self.input_shape)
        out = EinopsRearrangeMapper(out, 'c h w -> 1 (c h w)')

        for perceptron in self.perceptrons:
            out = PerceptronMapper(out, perceptron)

        return ModelRepresentation(out)

    def forward(self, x):
        out = self.input_activation(x)
        out = out.view(out.shape[0], -1)
        
        for perceptron in self.perceptrons:
            out = perceptron(out)

        return out