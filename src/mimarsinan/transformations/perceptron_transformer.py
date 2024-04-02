import torch
import torch.nn as nn

class PerceptronTransformer:
    def __init__(self):
        pass
    
    def get_effective_parameters(self, perceptron):
        return self.get_effective_weight(perceptron), self.get_effective_bias(perceptron)
    
    def apply_effective_parameter_transform(self, perceptron, parameter_transform):
        self.apply_effective_weight_transform(perceptron, parameter_transform)
        self.apply_effective_bias_transform(perceptron, parameter_transform)

    def get_effective_weight(self, perceptron):
        if isinstance(perceptron.normalization, nn.Identity):
            return perceptron.layer.weight.data
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            return perceptron.layer.weight.data * u.unsqueeze(-1)
        
    def get_effective_bias(self, perceptron):
        if perceptron.layer.bias is None:
            layer_bias = torch.zeros(perceptron.layer.weight.shape[0])
        else:
            layer_bias = perceptron.layer.bias.data

        if isinstance(perceptron.normalization, nn.Identity):
            return layer_bias
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            return (layer_bias - mean) * u + beta
        
    def apply_effective_weight_transform(self, perceptron, weight_transform):
        effective_weight = self.get_effective_weight(perceptron)
        
        if isinstance(perceptron.normalization, nn.Identity):
            perceptron.layer.weight.data[:] = weight_transform(effective_weight)
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            perceptron.layer.weight.data[:] = weight_transform(effective_weight) / u.unsqueeze(1)

    def apply_effective_bias_transform(self, perceptron, bias_transform):
        effective_bias = self.get_effective_bias(perceptron)
        
        if isinstance(perceptron.normalization, nn.Identity):
            perceptron.layer.bias.data[:] = bias_transform(effective_bias)
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            perceptron.layer.bias.data[:] = ((bias_transform(effective_bias) - beta) / u) + mean

    def _get_u_beta_mean(self, bn_layer):
        bn = bn_layer
        gamma = bn.weight.data
        beta = bn.bias.data
        var = bn.running_var.data.to(gamma.device)
        mean = bn.running_mean.data.to(gamma.device)

        u = gamma / torch.sqrt(var + bn.eps)

        return u, beta, mean