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

        self.out = None

    def get_perceptrons(self):
        return self.patch_layers + self.fc_layers + [self.output_layer]

    def fuse_normalization(self):
        for layer in self.get_perceptrons():
            layer.fuse_normalization()

    def set_activation(self, activation):
        for layer in self.get_perceptrons():
            layer.set_activation(activation)
    
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
        out_tensor = torch.zeros((x.shape[0], self.features))
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
    
    def statistical_consistency_loss(self, indicators):
        mean_of_indicators = indicators.mean()
        return torch.sqrt(torch.mean((indicators - mean_of_indicators) ** 2))
    
    def get_parameter_stats(self, parameter_list):
        param_means = []
        param_vars = []
        for layer in parameter_list:
            for param in layer.parameters():
                param_means.append(param.mean())
                param_vars.append(param.var())
        param_means = torch.stack(param_means)
        param_vars = torch.stack(param_vars)
        return param_means, param_vars

    def loss(self, x, y):
        stats = {}
        out = self(x, stats)

        param_means, param_vars = self.get_parameter_stats(self.fc_layers)
        param_mean_error = self.statistical_consistency_loss(param_means)
        param_var_error = self.statistical_consistency_loss(param_vars)

        param_means_patch, param_vars_patch = self.get_parameter_stats(self.patch_layers)
        param_mean_error_patch = self.statistical_consistency_loss(param_means_patch)
        param_var_error_patch = self.statistical_consistency_loss(param_vars_patch)

        mean_fc_error = self.statistical_consistency_loss(stats["means_fc"])
        var_fc_error = self.statistical_consistency_loss(stats["vars_fc"])
        mean_patch_error = self.statistical_consistency_loss(stats["means_patch"])
        var_patch_error = self.statistical_consistency_loss(stats["vars_patch"])

        activation_consistency_error = \
            torch.sqrt(torch.mean(torch.stack([
                mean_fc_error, var_fc_error, 
                mean_patch_error, var_patch_error]) ** 2))
        
        parameter_consistency_error = \
            torch.sqrt(torch.mean(torch.stack([
                param_mean_error, param_var_error,
                param_mean_error_patch, param_var_error_patch]) ** 2))
        

        self.out = out
        return nn.CrossEntropyLoss()(out, y) * (activation_consistency_error + parameter_consistency_error)
    

