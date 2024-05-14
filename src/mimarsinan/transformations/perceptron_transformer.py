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
            return (perceptron.layer.weight.data * u.unsqueeze(-1))
        
    def get_effective_bias(self, perceptron):
        if perceptron.layer.bias is None:
            layer_bias = torch.zeros(perceptron.layer.weight.shape[0])
        else:
            layer_bias = perceptron.layer.bias.data

        if isinstance(perceptron.normalization, nn.Identity):
            return layer_bias / perceptron.scale_factor
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            return ((layer_bias - mean) * u + beta) / perceptron.scale_factor
        
    def apply_effective_weight_transform(self, perceptron, weight_transform):
        effective_weight = self.get_effective_weight(perceptron)
        
        if isinstance(perceptron.normalization, nn.Identity):
            perceptron.layer.weight.data[:] = weight_transform(effective_weight)
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            perceptron.layer.weight.data[:] = (weight_transform(effective_weight) / u.unsqueeze(-1))

    def apply_effective_bias_transform(self, perceptron, bias_transform):
        effective_bias = self.get_effective_bias(perceptron)
        
        if isinstance(perceptron.normalization, nn.Identity):
            perceptron.layer.bias.data[:] = bias_transform(effective_bias) * perceptron.scale_factor
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            perceptron.layer.bias.data[:] = (((bias_transform(effective_bias) - beta * perceptron.scale_factor) / u) + mean)


    def apply_effective_bias_transform_to_norm(self, perceptron, bias_transform):
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
    
# test code 

# if run as main
if __name__ == "__main__":
    class Perceptron(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.layer = nn.Linear(input_size, output_size)
            self.normalization = nn.BatchNorm1d(output_size)
            self.scale_factor = 0.3169
            self.act = nn.LeakyReLU()
        
        def forward(self, x):
            out = self.layer(x)
            out = self.normalization(out)
            return self.act(out)
        

    with torch.no_grad():
        perceptron_transformer = PerceptronTransformer()

        # test case 1 : with normalization
        print("TEST CASE")
        perceptron = Perceptron(10, 5)
        perceptron.eval()

        rand_input = torch.rand(size=(1, 10))

        # non-fused
        perceptron.layer.weight.data = torch.randn_like(perceptron.layer.weight.data)
        perceptron.layer.bias.data = torch.randn_like(perceptron.layer.bias.data)
        perceptron.normalization.weight.data = torch.randn_like(perceptron.normalization.weight.data)
        perceptron.normalization.bias.data = torch.randn_like(perceptron.normalization.bias.data)
        perceptron.normalization.running_mean.data = torch.randn_like(perceptron.normalization.running_mean.data)
        perceptron.normalization.running_var.data = torch.abs(torch.randn_like(perceptron.normalization.running_var.data))

        perceptron_transformer.apply_effective_parameter_transform(perceptron, lambda x: x * 0.42 + 0.69)
        out_1 = perceptron(rand_input)


        # fused
        w, b = perceptron_transformer.get_effective_parameters(perceptron)
        perceptron.layer.weight.data = w
        perceptron.layer.bias.data = b
        perceptron.normalization = nn.Identity()
        inp = rand_input / perceptron.scale_factor
        out_2 =  perceptron(inp) * perceptron.scale_factor

        # test
        print(out_1)
        print(out_2)


        # test case : with normalization transform norm
        print("TEST CASE")
        perceptron = Perceptron(10, 5)
        perceptron.eval()

        rand_input = torch.rand(size=(1, 10))


        # non-fused
        perceptron.layer.weight.data = torch.randn_like(perceptron.layer.weight.data)
        perceptron.layer.bias.data = torch.randn_like(perceptron.layer.bias.data)
        perceptron.normalization.weight.data = torch.randn_like(perceptron.normalization.weight.data)
        perceptron.normalization.bias.data = torch.randn_like(perceptron.normalization.bias.data)
        perceptron.normalization.running_mean.data = torch.randn_like(perceptron.normalization.running_mean.data)
        perceptron.normalization.running_var.data = torch.abs(torch.randn_like(perceptron.normalization.running_var.data))

        perceptron_transformer.apply_effective_bias_transform_to_norm(perceptron, lambda x: x * 0.42 + 0.69)
        out_1 = perceptron(rand_input)


        # fused
        w, b = perceptron_transformer.get_effective_parameters(perceptron)
        perceptron.layer.weight.data = w
        perceptron.layer.bias.data = b
        perceptron.normalization = nn.Identity()
        inp = rand_input / perceptron.scale_factor
        out_2 =  perceptron(inp) * perceptron.scale_factor

        # test
        print(out_1)
        print(out_2)

        ###
        # test case 2 : without normalization
        print("TEST CASE")
        perceptron = Perceptron(10, 5)
        perceptron.normalization = nn.Identity()
        perceptron.eval()

        # non-fused
        rand_input = torch.rand(size=(1, 10))
        perceptron.layer.weight.data = torch.randn_like(perceptron.layer.weight.data)
        perceptron.layer.bias.data = torch.randn_like(perceptron.layer.bias.data)
        perceptron_transformer.apply_effective_parameter_transform(perceptron, lambda x: x * 0.42 + 0.69)
        out_1 = perceptron(rand_input)

        # fused
        w, b = perceptron_transformer.get_effective_parameters(perceptron)
        perceptron.layer.weight.data = w
        perceptron.layer.bias.data = b
        perceptron.normalization = nn.Identity()
        inp = rand_input / perceptron.scale_factor
        out_2 = perceptron(inp) * perceptron.scale_factor

        # test
        print(out_1)
        print(out_2)


        # test q
        print("TEST CASE")
        perceptron = Perceptron(10, 5)
        perceptron.layer.weight.data = torch.randn_like(perceptron.layer.weight.data)
        perceptron.layer.bias.data = torch.randn_like(perceptron.layer.bias.data)
        perceptron.normalization.weight.data = torch.randn_like(perceptron.normalization.weight.data)
        perceptron.normalization.bias.data = torch.randn_like(perceptron.normalization.bias.data)
        perceptron.normalization.running_mean.data = torch.randn_like(perceptron.normalization.running_mean.data)
        perceptron.normalization.running_var.data = torch.abs(torch.randn_like(perceptron.normalization.running_var.data))
        perceptron.eval()

        w = PerceptronTransformer().get_effective_weight(perceptron)
        b = PerceptronTransformer().get_effective_bias(perceptron)
        
        w_max = torch.max(torch.abs(w))
        b_max = torch.max(torch.abs(b))
        p_max = max(w_max, b_max)

        scale = 127 * (1.0 / p_max)

        # do magick here:
        # scale_09 = scale * 0.99
        # scale_09_r = max(torch.floor(scale_09), 1.0)
        # scale = scale_09_r / 0.99 # end magick

        # scale = torch.round(scale)
        def quantize_param(param):
            scaled_param = param * scale
            quantized_param = torch.round(scaled_param)
            quantized_param = torch.minimum(quantized_param, torch.tensor(127))
            quantized_param = torch.maximum(quantized_param, torch.tensor(-128))
            rescaled_param = quantized_param / (scale)
            return rescaled_param
        
        PerceptronTransformer().apply_effective_parameter_transform(perceptron, quantize_param) 
        out_1 = perceptron(rand_input)

        w, b = PerceptronTransformer().get_effective_parameters(perceptron)
        perceptron.layer.weight.data = w
        perceptron.layer.bias.data = b
        perceptron.normalization = nn.Identity()

        inp = rand_input / perceptron.scale_factor
        out_2 = perceptron(inp) * perceptron.scale_factor

        print(out_1)
        print(out_2)



