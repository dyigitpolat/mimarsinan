from mimarsinan.mapping.mapping_utils import *

import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(
        self, 
        output_channels, input_features, bias=True,
        normalization = nn.Identity()):

        super(Perceptron, self).__init__()
        
        self.layer = nn.Linear(
            input_features, output_channels, bias=bias)

        self.normalization = normalization
        self.activation = nn.LeakyReLU()
    
    def set_activation(self, activation):
        self.activation = activation

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

        l = self.perceptron.layer
        bn = self.perceptron.normalization
    
        w = l.weight.data
        b = l.bias.data if l.bias is not None else torch.zeros(w.shape[0])

        new_w = l.weight.data.clone()
        new_b = l.bias.data.clone()

        gamma = bn.weight.data
        beta = bn.bias.data
        var = bn.running_var.data
        mean = bn.running_mean.data
        u = gamma / torch.sqrt(var + bn.eps)

        new_w[:,:] = w * u.unsqueeze(1)
        new_b[:] = (b - mean) * u + beta
        
        return new_w, new_b

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
            Perceptron(256, 256),
            Perceptron(256, 256, normalization=nn.BatchNorm1d(256)),
            Perceptron(256, 256, normalization=nn.BatchNorm1d(256)),
            Perceptron(num_classes, 256)
        ])
    
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        for layer in self.layers:
            out = layer(out)
        
        return out
    
    def get_mapper_repr(self):
        out = InputMapper(self.input_shape)
        for layer in self.layers:
            out = PerceptronMapper(out, layer)
        
        return ModelRepresentation(out)

