from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow

import torch.nn as nn
import einops

class PerceptronMixer(PerceptronFlow):
    def __init__(
        self, 
        input_shape,
        num_classes,
        max_axons,
        max_neurons, 
        patch_width,
        patch_height,
        fc_depth=1):
        super(PerceptronMixer, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_shape[-3]
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_size = self.patch_width * self.patch_height * self.input_channels
        self.patch_rows = input_shape[-2] // self.patch_height
        self.patch_cols = input_shape[-1] // self.patch_width
        self.patch_count = self.patch_rows * self.patch_cols

        self.features = min(max_axons, max_neurons)
        self.patch_channels = self.features // self.patch_count
        self.fc_width = self.patch_channels * self.patch_count

        assert self.fc_width <= max_neurons, f"not enough neurons ({self.fc_width} > {max_neurons})"
        assert self.fc_width <= max_axons, f"not enough axons ({self.fc_width} > {max_axons})"
        assert self.patch_size <= max_axons, f"not enough axons ({self.patch_size} > {max_axons})"

        self.patch_layers = nn.ModuleList(
            [Perceptron(self.patch_channels, self.patch_size) for _ in range(self.patch_count)])
        
        self.fc_layers = nn.ModuleList()
        for _ in range(fc_depth):
            self.fc_layers.append(Perceptron(self.fc_width, self.fc_width))
            self.fc_layers.append(Perceptron(self.fc_width, self.fc_width, 
                                             normalization=nn.BatchNorm1d(self.fc_width)))

        self.output_layer = Perceptron(num_classes, self.fc_width) 
        self.activation = nn.LeakyReLU()

        self.out = None

    def get_perceptrons(self):
        return self.patch_layers + self.fc_layers + [self.output_layer]
    
    def get_mapper_repr(self):
        out = InputMapper(self.input_shape)
        out = EinopsRearrangeMapper(
            out, 
            'c (h p1) (w p2) -> (h w) (p1 p2 c)', 
            p1=self.patch_height, p2=self.patch_width)
        
        patch_mappers = []
        for idx in range(self.patch_count):
            source = SubscriptMapper(out, idx)
            source = ReshapeMapper(source, (1, self.patch_size))
            patch_mappers.append(PerceptronMapper(source, self.patch_layers[idx]))
        out = StackMapper(patch_mappers)
        out = ReshapeMapper(out, (1, self.patch_count * self.patch_channels))

        for layer in self.fc_layers:
            out = PerceptronMapper(out, layer)
        
        out = PerceptronMapper(out, self.output_layer)
        return ModelRepresentation(out)

    def forward(self, x, stats={}):
        out = einops.einops.rearrange(
            x, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=self.patch_height, p2=self.patch_width)
        
        means_patch, vars_patch = [], []
        out_tensor = torch.zeros((x.shape[0], self.fc_width), device=x.device)
        for idx in range(self.patch_count):
            length = self.patch_channels
            out_tensor[:, idx*length:(idx+1)*length] = self.patch_layers[idx](out[:,idx])
            means_patch.append(out_tensor[:, idx*length:(idx+1)*length].flatten().mean())
            vars_patch.append(out_tensor[:, idx*length:(idx+1)*length].flatten().var())
        stats["means_patch"] = torch.stack(means_patch)
        stats["vars_patch"] = torch.stack(vars_patch)

        means_fc, vars_fc = [], []
        out = out_tensor
        for layer in self.fc_layers:
            out = layer(out)
            means_fc.append(out.flatten().mean())
            vars_fc.append(out.flatten().var())
        stats["means_fc"] = torch.stack(means_fc)
        stats["vars_fc"] = torch.stack(vars_fc)

        out = self.output_layer(out)
        return out