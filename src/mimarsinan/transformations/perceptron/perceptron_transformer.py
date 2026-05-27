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

    def _get_input_scale(self, perceptron):
        """Return broadcastable input scale: per-channel tensor or scalar 1.0."""
        pis = getattr(perceptron, 'per_input_scales', None)
        if pis is not None:
            pis = pis.to(perceptron.layer.weight.data.device)
            extra = perceptron.layer.weight.data.dim() - 2  # 0 for FC, 2 for Conv2D
            return pis.view(1, -1, *([1] * extra))
        return 1.0

    def get_effective_weight(self, perceptron):
        scale = self._get_input_scale(perceptron)
        if isinstance(perceptron.normalization, nn.Identity):
            return scale * perceptron.layer.weight.data / perceptron.activation_scale
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            return scale * (perceptron.layer.weight.data * u.unsqueeze(-1)) / perceptron.activation_scale
        
    def get_effective_bias(self, perceptron):
        if perceptron.layer.bias is None:
            layer_bias = torch.zeros(perceptron.layer.weight.shape[0])
        else:
            layer_bias = perceptron.layer.bias.data

        if isinstance(perceptron.normalization, nn.Identity):
            return layer_bias / perceptron.activation_scale
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            return ((layer_bias - mean) * u + beta) / perceptron.activation_scale
        
    def apply_effective_weight_transform(self, perceptron, weight_transform):
        effective_weight = self.get_effective_weight(perceptron)
        scale = self._get_input_scale(perceptron)
        
        if isinstance(perceptron.normalization, nn.Identity):
            perceptron.layer.weight.data[:] = (weight_transform(effective_weight) * perceptron.activation_scale) / scale
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            perceptron.layer.weight.data[:] = ((weight_transform(effective_weight) * perceptron.activation_scale) / scale) / u.unsqueeze(-1)

    def apply_effective_bias_transform(self, perceptron, bias_transform):
        if perceptron.layer.bias is None:
            return
        effective_bias = self.get_effective_bias(perceptron)
        
        if isinstance(perceptron.normalization, nn.Identity):
            perceptron.layer.bias.data[:] = bias_transform(effective_bias) * perceptron.activation_scale
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            perceptron.layer.bias.data[:] = (((bias_transform(effective_bias) * perceptron.activation_scale - beta) / u) + mean)


    def apply_effective_bias_transform_to_norm(self, perceptron, bias_transform):
        if perceptron.layer.bias is None:
            return
        effective_bias = self.get_effective_bias(perceptron)
        
        if isinstance(perceptron.normalization, nn.Identity):
            perceptron.layer.bias.data[:] = bias_transform(effective_bias) * perceptron.scale_factor
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            perceptron.normalization.running_mean.data[:] = bias_transform(mean)

    def _get_u_beta_mean(self, bn_layer):
        bn = bn_layer
        gamma = bn.weight.data
        beta = bn.bias.data
        var = bn.running_var.data.to(gamma.device)
        mean = bn.running_mean.data.to(gamma.device)

        u = gamma / torch.sqrt(var + bn.eps)

        return u, beta, mean
