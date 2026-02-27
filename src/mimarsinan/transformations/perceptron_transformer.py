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
        effective_bias = self.get_effective_bias(perceptron)
        
        if isinstance(perceptron.normalization, nn.Identity):
            perceptron.layer.bias.data[:] = bias_transform(effective_bias) * perceptron.activation_scale
        else:
            u, beta, mean = self._get_u_beta_mean(perceptron.normalization)
            perceptron.layer.bias.data[:] = (((bias_transform(effective_bias) * perceptron.activation_scale - beta) / u) + mean)


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
            self.shift_amt = 0.0
            self.act = nn.LeakyReLU()
        
        def forward(self, x):
            out = self.layer(x)
            print(out.shape)
            out = self.normalization(out)
            return self.act(out + self.shift_amt)
        

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
        print("TEST CASE with norm, and transform norm")
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

        perceptron_transformer.apply_effective_weight_transform(perceptron, lambda x: x * 0.42 + 0.69)
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

        ###
        # test case 2 : without normalization apply to norm
        print("TEST CASE without norm apply to norm")
        perceptron = Perceptron(10, 5)
        perceptron.normalization = nn.Identity()
        perceptron.eval()

        # non-fused
        rand_input = torch.rand(size=(1, 10))
        perceptron.layer.weight.data = torch.randn_like(perceptron.layer.weight.data)
        perceptron.layer.bias.data = torch.randn_like(perceptron.layer.bias.data)

        perceptron_transformer.apply_effective_weight_transform(perceptron, lambda x: x * 0.42 + 0.69)
        perceptron_transformer.apply_effective_bias_transform_to_norm(perceptron, lambda x: x * 0.42 + 0.69)
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

        # shift test 
        print("TEST CASE shift")
        perceptron = Perceptron(10, 5)
        perceptron.normalization = nn.Identity()
        perceptron.scale_factor = 1.0
        perceptron.eval()

        rand_input = torch.rand(size=(1, 10))

        # non-fused
        perceptron.layer.weight.data = torch.randn_like(perceptron.layer.weight.data)
        perceptron.layer.bias.data = torch.randn_like(perceptron.layer.bias.data)

        out_1 = perceptron(rand_input)

        shift_amt = 0.9876
        perceptron.shift_amt = -shift_amt
        perceptron.layer.bias.data[:] = perceptron.layer.bias.data[:] + shift_amt

        out_2 = perceptron(rand_input)

        print(out_1)
        print(out_2)

        # shift test norm
        print("TEST CASE shift norm")
        perceptron = Perceptron(10, 5)
        perceptron.scale_factor = 1.0
        perceptron.eval()

        rand_input = torch.rand(size=(1, 10))

        # non-fused
        perceptron.layer.weight.data = torch.randn_like(perceptron.layer.weight.data)
        perceptron.layer.bias.data = torch.randn_like(perceptron.layer.bias.data)
        perceptron.normalization.weight.data = torch.randn_like(perceptron.normalization.weight.data)
        perceptron.normalization.bias.data = torch.randn_like(perceptron.normalization.bias.data)
        perceptron.normalization.running_mean.data = torch.randn_like(perceptron.normalization.running_mean.data)
        perceptron.normalization.running_var.data = torch.abs(torch.randn_like(perceptron.normalization.running_var.data))

        out_1 = perceptron(rand_input)

        shift_amt = 0.9876
        perceptron.shift_amt = -shift_amt
        PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda b: b + shift_amt)

        out_3 = perceptron(rand_input)


        w, b = PerceptronTransformer().get_effective_parameters(perceptron)
        perceptron.layer.weight.data = w
        perceptron.layer.bias.data = b
        perceptron.normalization = nn.Identity()
        
        out_2 = perceptron(rand_input)

        print(out_1)
        print(out_2)
        print(out_3)



        # adjust norm test func
        print("TEST CASE adjust norm test func")
        def _adjust_normalization_stats(norm, scale):
            
            # Adjust running mean
            norm.running_mean.data[:] *= scale
            
            # Adjust running variance
            norm.running_var.data[:] *= scale**2

        norm = nn.BatchNorm1d(5)
        norm.weight.data = torch.randn_like(norm.weight.data)
        norm.bias.data = torch.randn_like(norm.bias.data)
        norm.running_mean.data = torch.randn_like(norm.running_mean.data)
        norm.running_var.data = torch.abs(torch.randn_like(norm.running_var.data))

        random_input = torch.randn(2, 5) + 5.0
        scale = 3

        old = torch.functional.F.batch_norm(random_input, norm.running_mean, norm.running_var, norm.weight, norm.bias, False, 0.0, norm.eps)
        _adjust_normalization_stats(norm, scale)
        new = torch.functional.F.batch_norm(random_input * scale, norm.running_mean, norm.running_var, norm.weight, norm.bias, False, 0.0, norm.eps)

        print("Old", old)
        print("New", new)


        # adjust norm test module
        print("TEST CASE adjust norm test func")
        random_input = torch.randn(2, 5) + 5.0
        scale = 3

        norm.eval()
        old = norm(random_input)
        _adjust_normalization_stats(norm, scale)

        norm.eval()
        new = norm(torch.tensor(random_input.numpy()* scale) )

        print("Old", old)
        print("New", new)


