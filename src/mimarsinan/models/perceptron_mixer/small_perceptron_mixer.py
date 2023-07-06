from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow

import torch.nn as nn
import einops

class SmallPerceptronMixer(PerceptronFlow):
    def __init__(
        self, 
        input_shape,
        num_classes,
        patch_n_1,
        patch_m_1,
        patch_c_1,
        fc_w_1,
        fc_k_1):
        super(SmallPerceptronMixer, self).__init__()
        
        self.input_shape = input_shape
        self.input_channels = input_shape[-3]

        self.patch_rows = patch_n_1
        self.patch_cols = patch_m_1
        self.patch_channels = patch_c_1

        self.patch_height = input_shape[-2] // self.patch_rows
        self.patch_width = input_shape[-1] // self.patch_cols

        self.patch_size = self.patch_height * self.patch_width * self.input_channels
        self.patch_count = self.patch_rows * self.patch_cols

        self.fc_in = self.patch_count * self.patch_channels
        self.fc_depth = fc_k_1
        self.fc_width = fc_w_1

        self.patch_layers = nn.ModuleList(
            [Perceptron(self.patch_channels, self.patch_size) for _ in range(self.patch_count)])
    
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(Perceptron(self.fc_width, self.fc_in, 
                                         normalization=nn.BatchNorm1d(self.fc_width)))
        for idx in range(self.fc_depth - 1):
            if idx % 2 == 0:
                norm = nn.Identity()
            else:
                norm = nn.BatchNorm1d(self.fc_width)
            self.fc_layers.append(Perceptron(self.fc_width, self.fc_width, 
                                             normalization=norm))

        self.output_layer = Perceptron(num_classes, self.fc_width) 
        self.activation = nn.LeakyReLU()

        self.out = None

    def get_perceptrons(self):
        return self.patch_layers + self.fc_layers + [self.output_layer]
    
    def get_mapper_repr(self):
        out = InputMapper(self.input_shape)

        # First Mixer
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
        
        # Output Layer
        out = PerceptronMapper(out, self.output_layer)
        return ModelRepresentation(out)

    def forward(self, x):
        # First Mixer
        out = einops.einops.rearrange(
            x, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=self.patch_height, p2=self.patch_width)
        
        out_tensor = torch.zeros((x.shape[0], self.fc_in), device=x.device)
        for idx in range(self.patch_count):
            length = self.patch_channels
            out_tensor[:, idx*length:(idx+1)*length] = self.patch_layers[idx](out[:,idx])

        out = out_tensor
        for layer in self.fc_layers:
            out = layer(out)

        out = self.output_layer(out)
        return out