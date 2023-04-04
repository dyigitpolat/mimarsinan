from mimarsinan.mapping.mapping_utils import *

import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(
        self, 
        output_channels, input_features, bias=True,
        normalization = nn.Identity()):

        super(Perceptron, self).__init__()
        
        self.input_features = input_features
        self.output_channels = output_channels

        self.layer = nn.Linear(
            input_features, output_channels, bias=bias)

        self.normalization = normalization
        self.activation = nn.LeakyReLU()
    
    def set_activation(self, activation):
        self.activation = activation

    def fuse_normalization(self):
        if isinstance(self.normalization, nn.Identity):
            return

        assert isinstance(self.normalization, nn.BatchNorm1d)
        assert self.normalization.affine

        w, b = get_fused_weights(
            linear_layer=self.layer, bn_layer=self.normalization)

        self.layer = nn.Linear(
            self.input_features, self.output_channels, bias=True)
        self.layer.weight.data = w
        self.layer.bias.data = b

        self.normalization = nn.Identity()

    def forward(self, x):
        out = self.layer(x)
        out = self.normalization(out)
        out = self.activation(out)
        
        return out


class PerceptronMapper:
    def __init__(self, source_mapper, perceptron):
        self.perceptron = perceptron
        self.source_mapper = source_mapper
        self.sources = None

    def fuse_normalization(self):
        if isinstance(self.perceptron.normalization, nn.Identity):
            layer = self.perceptron.layer
            w = layer.weight.data
            b = layer.bias.data if layer.bias is not None else None
            return w, b
        
        assert isinstance(self.perceptron.normalization, nn.BatchNorm1d)
        assert self.perceptron.normalization.affine

        return get_fused_weights(
            linear_layer=self.perceptron.layer, 
            bn_layer=self.perceptron.normalization)

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_weights, layer_biases = self.fuse_normalization()
        layer_weights.detach().numpy()
        layer_biases.detach().numpy()

        layer_sources = self.source_mapper.map(mapping)
        layer_sources = layer_sources.transpose()
        layer_sources = map_mm(mapping, layer_sources, layer_weights, layer_biases)
        layer_sources = layer_sources.transpose()

        self.sources = layer_sources
        return self.sources
    

class SimplePerceptronFlow(nn.Module):
    def __init__(
        self, 
        input_shape,
        num_classes):
        super(SimplePerceptronFlow, self).__init__()
        
        self.input_shape = input_shape
        self.layers = nn.ModuleList([
            Perceptron(256, input_shape[1]),
            Perceptron(256, 256),
            Perceptron(256, 256, normalization=nn.BatchNorm1d(256)),
            Perceptron(256, 256),
            Perceptron(256, 256, normalization=nn.BatchNorm1d(256)),
            Perceptron(256, 256),
            Perceptron(256, 256, normalization=nn.BatchNorm1d(256)),
            Perceptron(num_classes, 256)
        ])
    
    
    def get_mapper_repr(self):
        out = InputMapper(self.input_shape)
        for layer in self.layers:
            out = PerceptronMapper(out, layer)
        
        return ModelRepresentation(out)
    
    def fuse_normalization(self):
        for layer in self.layers:
            layer.fuse_normalization()

    def set_activation(self, activation):
        for layer in self.layers:
            layer.set_activation(activation)
    
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        for layer in self.layers:
            out = layer(out)
        
        return out
    
import einops
class PatchedPerceptronFlow(nn.Module):
    def __init__(
        self, 
        input_shape,
        num_classes,
        max_axons,
        max_neurons, 
        patch_width,
        patch_height,
        fc_depth=1):
        super(PatchedPerceptronFlow, self).__init__()

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
        self.patch_layers = nn.ModuleList(
            [Perceptron(self.patch_channels, self.patch_size) for _ in range(self.patch_count)])
        
        self.fc_layers = nn.ModuleList()
        for _ in range(fc_depth):
            self.fc_layers.append(Perceptron(self.features, self.features))
            self.fc_layers.append(Perceptron(self.features, self.features, 
                                             normalization=nn.BatchNorm1d(self.features)))

        self.output_layer = Perceptron(num_classes, self.features) 

    def fuse_normalization(self):
        for layer in self.patch_layers:
            layer.fuse_normalization()

        for layer in self.fc_layers:
            layer.fuse_normalization()

        self.output_layer.fuse_normalization()

    def set_activation(self, activation):
        for layer in self.patch_layers:
            layer.set_activation(activation)

        for layer in self.fc_layers:
            layer.set_activation(activation)

        self.output_layer.set_activation(activation)
    
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

    def forward(self, x):
        out = einops.einops.rearrange(
            x, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=self.patch_height, p2=self.patch_width)
        
        out_tensor = torch.zeros((x.shape[0], self.features))
        for idx in range(self.patch_count):
            length = self.patch_channels
            out_tensor[:, idx*length:(idx+1)*length] = self.patch_layers[idx](out[:,idx])
        
        out = out_tensor
        for layer in self.fc_layers:
            out = layer(out)

        out = self.output_layer(out)
        
        return out