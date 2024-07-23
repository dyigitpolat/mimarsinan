from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow

import torch.nn as nn
import einops

class PerceptronMixer(PerceptronFlow):
    def __init__(
        self, 
        device,
        input_shape,
        num_classes,
        patch_n_1,
        patch_m_1,
        patch_c_1,
        fc_w_1,
        fc_k_1,
        patch_n_2,
        patch_c_2,
        fc_w_2,
        fc_k_2):
        super(PerceptronMixer, self).__init__(device)

        self.input_activation = nn.Identity()
        
        self.input_shape = input_shape
        self.input_channels = input_shape[-3]

        self.patch_rows = patch_n_1
        self.patch_cols = patch_m_1
        self.patch_channels = patch_c_1

        self.fc_depth = fc_k_1
        self.fc_depth_2 = fc_k_2

        self.patch_height = input_shape[-2] // self.patch_rows
        self.patch_width = input_shape[-1] // self.patch_cols

        self.patch_size = self.patch_height * self.patch_width * self.input_channels
        self.patch_count = self.patch_rows * self.patch_cols


        self.patch_layer = Perceptron(self.patch_channels, self.patch_size, normalization=nn.LazyBatchNorm1d() ) 
        self.patch_layer_CONV = nn.Conv2d(self.input_channels, self.patch_channels, kernel_size=self.patch_height, stride=self.patch_height)

        self.patch_layers_list = nn.ModuleList()
        self.patch_layers_list_2 = nn.ModuleList()
        self.fc_layers_list = nn.ModuleList()
        self.fc_layers_list_2 = nn.ModuleList()

        self.mixer_count = 2

        XX = fc_w_1
        YY = fc_w_2
        for mixer_idx in range(self.mixer_count):
            self.patch_layers_list.append(Perceptron(XX, self.patch_count, normalization=nn.LazyBatchNorm1d(), name="tok_mixer_{}".format(mixer_idx)))
            self.fc_layers_list.append(Perceptron(self.patch_count, XX))

            self.patch_layers_list_2.append(Perceptron(YY, self.patch_channels, normalization=nn.LazyBatchNorm1d(), name="ch_mixer_{}".format(mixer_idx)))
            self.fc_layers_list_2.append(Perceptron(self.patch_channels, YY))
        
        self.output_layer = Perceptron(num_classes, self.patch_count * self.patch_channels)

        self.out = None

    def get_input_activation(self):
        return self.input_activation
    
    def set_input_activation(self, activation):
        self.input_activation = activation

    def get_perceptrons(self):
        perceptrons = []
        perceptrons += [self.patch_layer]
        perceptrons += self.patch_layers_list
        perceptrons += self.fc_layers_list
        perceptrons += self.patch_layers_list_2
        perceptrons += self.fc_layers_list_2
        perceptrons.append(self.output_layer)

        return perceptrons
    
    def get_perceptron_groups(self):
        groups = []
        groups.append([self.patch_layer])
        for idx in range(self.mixer_count):
            groups.append([self.patch_layers_list[idx]])
            groups.append([self.fc_layers_list[idx]])
            groups.append([self.patch_layers_list_2[idx]])
            groups.append([self.fc_layers_list_2[idx]])
        groups.append([self.output_layer])

        return groups
    
    def get_mapper_repr(self):
        out = InputMapper(self.input_shape)

        # Patcher:
        out = EinopsRearrangeMapper(
            out, 
            'c (h p1) (w p2) -> (h w) (p1 p2 c)', 
            p1=self.patch_height, p2=self.patch_width)
        
        patch_mappers = []
        for idx in range(self.patch_count):
            source = SubscriptMapper(out, idx)
            source = ReshapeMapper(source, (1, self.patch_size))
            patch_mappers.append(PerceptronMapper(source, self.patch_layer))
        out = StackMapper(patch_mappers)
        out = ReshapeMapper(out, (1, self.patch_count * self.patch_channels))

        out = EinopsRearrangeMapper(
            out, 
            '1 (np cp) -> cp np', 
            np=self.patch_count, cp=self.patch_channels)
        
        for mixer_idx in range(self.mixer_count):
            # Token Mixer
            patch_mappers = []
            for idx in range(self.patch_channels):
                source = SubscriptMapper(out, idx)
                source = ReshapeMapper(source, (1, self.patch_count))

                res_source = source
                source = PerceptronMapper(source, self.patch_layers_list[mixer_idx])
                source = PerceptronMapper(source, self.fc_layers_list[mixer_idx])
                #patch_mappers.append(AddMapper(source, DelayMapper(res_source, 1))) #res
                patch_mappers.append(source)

            out = StackMapper(patch_mappers)
            out = ReshapeMapper(out, (1, self.patch_count * self.patch_channels))
            out = EinopsRearrangeMapper(
                out, 
                '1 (cp np) -> np cp', 
                np=self.patch_count, cp=self.patch_channels)
            
            # Channel Mixer
            patch_mappers = []
            for idx in range(self.patch_count):
                source = SubscriptMapper(out, idx)
                source = ReshapeMapper(source, (1, self.patch_channels))
                
                res_source = source
                source = PerceptronMapper(source, self.patch_layers_list_2[mixer_idx])
                source = PerceptronMapper(source, self.fc_layers_list_2[mixer_idx])
                #patch_mappers.append(AddMapper(source, DelayMapper(res_source, 1))) #res
                patch_mappers.append(source)

            out = StackMapper(patch_mappers)
            out = ReshapeMapper(out, (1, self.patch_count * self.patch_channels))
            out = EinopsRearrangeMapper(
                out, 
                '1 (np cp) -> cp np', 
                np=self.patch_count, cp=self.patch_channels)
        
        # Output Layer
        out = EinopsRearrangeMapper(
            out, 
            'cp np -> 1 (np cp)',
            np=self.patch_count, cp=self.patch_channels)
        out = PerceptronMapper(out, self.output_layer)
        return ModelRepresentation(out)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.input_activation(x)

        # Patcher:
        out = einops.einops.rearrange(
            out, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=self.patch_height, p2=self.patch_width)
        
        # Patchwise MLP
        out = einops.einops.rearrange(out, 'b np ps -> (b np) ps', np=self.patch_count, ps=self.patch_size)
        out = self.patch_layer(out)
        out = einops.einops.rearrange(out, '(b np) cp -> b np cp', np=self.patch_count, cp=self.patch_channels)
        
        for mixer_idx in range(self.mixer_count):
            # Token Mixer
            out = einops.einops.rearrange(out, 'b np cp -> b cp np', np=self.patch_count, cp=self.patch_channels)
            out = einops.einops.rearrange(out, 'b cp np -> (b cp) np', np=self.patch_count, cp=self.patch_channels)
            res = out
            out = self.patch_layers_list[mixer_idx](out)
            out = self.fc_layers_list[mixer_idx](out) #+ res
            out = einops.einops.rearrange(out, '(b cp) np -> b cp np', np=self.patch_count, cp=self.patch_channels)

            # Channel Mixer
            out = einops.einops.rearrange(out, 'b cp np -> b np cp', np=self.patch_count, cp=self.patch_channels)
            out = einops.einops.rearrange(out, 'b np cp -> (b np) cp', np=self.patch_count, cp=self.patch_channels)
            res = out
            out = self.patch_layers_list_2[mixer_idx](out)
            out = self.fc_layers_list_2[mixer_idx](out) #+ res
            out = einops.einops.rearrange(out, '(b np) cp -> b np cp', np=self.patch_count, cp=self.patch_channels)
        
        out = einops.einops.rearrange(out, 'b np cp -> b (np cp)', np=self.patch_count, cp=self.patch_channels)
        out = self.output_layer(out)
        return out