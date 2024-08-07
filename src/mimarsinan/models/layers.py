from torch import Tensor
import torch.nn as nn
import torch
    
from torch.autograd import Function

class LeakyGradReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, negative_slope=1e-8):
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        return torch.where(input > 0, input, 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] *= ctx.negative_slope
        return grad_input, None

class LeakyGradReLU(nn.Module): 
    def __init__(self, negative_slope=1e-8):
        super(LeakyGradReLU, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        return LeakyGradReLUFunction.apply(x, self.negative_slope)

class StaircaseFunction(Function):
    @staticmethod
    def forward(ctx, x, Tq):
        #Tq = torch.tensor(Tq)
        #ctx.save_for_backward(x, Tq)
        return torch.floor(x * Tq) / Tq

    @staticmethod
    def backward(ctx, grad_output):
        #x, Tq = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None

class DifferentiableClamp(Function):
    @staticmethod
    def forward(ctx, x, a, b):
        a = a.clone().detach().to(x.device)
        b = b.clone().detach().to(x.device)
    
        ctx.save_for_backward(x, a, b)
        return torch.clamp(x, a, b)

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b = ctx.saved_tensors
        grad = torch.where(
            x < a,
            torch.exp(x-a),
            torch.where(
                x < b,
                1.0,
                torch.exp(b-x)))
    
        grad_input = grad_output * grad
        return grad_input, None, None
    
class NoisyDropout(nn.Module):
    def __init__(self, dropout_p, rate, noise_radius):
        super(NoisyDropout, self).__init__()
        self.dropout_p = dropout_p
        self.rate = rate
        self.noise_radius = noise_radius
    
    def forward(self, x):
        random_mask = torch.rand(x.shape, device=x.device)
        random_mask = (random_mask < self.rate).float()

        out = nn.Dropout(self.dropout_p)(x)
        out = out + self.noise_radius * torch.rand_like(out) - 0.5 * self.noise_radius
        return random_mask * out + (1.0 - random_mask) * x

class NoiseDecorator:
    def __init__(self, rate, noise_radius):
        self.rate = rate
        self.noise_radius = noise_radius
    
    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        return NoisyDropout(torch.tensor(0.0), self.rate, self.noise_radius)(x)
    
class SavedTensorDecorator(nn.Module):
    def __init__(self):
        super(SavedTensorDecorator, self).__init__()
        self.latest_input = None
        self.latest_output = None
    
    def input_transform(self, x):
        if(len(x.shape) > 1):
            self.latest_input = x
            
        return nn.Identity()(x)
    
    def output_transform(self, x):
        if(len(x.shape) > 1):
            self.latest_output = x

        return nn.Identity()(x)
    
class StatsDecorator:
    def __init__(self):
        self.in_mean = None
        self.in_var = None
        self.in_max = None
        self.in_min = None

        self.in_hist = None
        self.in_hist_bin_edges = None

        self.out_mean = None
        self.out_var = None
        self.out_max = None
        self.out_min = None

        self.out_hist = None
        self.out_hist_bin_edges = None
    
    def input_transform(self, x):


        if(len(x.shape) > 1):
            self.in_mean = torch.mean(x)
            self.in_var = torch.var(x)
            self.in_max = torch.max(x)
            self.in_min = torch.min(x)

            self.in_hist = torch.histc(x.flatten(), bins=100, min=self.in_min.item(), max=self.in_max.item())
            self.in_hist_bin_edges = torch.linspace(self.in_min.item(), self.in_max.item(), steps=101)

        return nn.Identity()(x)
    
    def output_transform(self, x):
        
        if(len(x.shape) > 1):
            self.out_mean = torch.mean(x)
            self.out_var = torch.var(x)
            self.out_max = torch.max(x)
            self.out_min = torch.min(x)

            self.out_hist = torch.histc(x.flatten(), bins=100, min=self.out_min.item(), max=self.out_max.item())
            self.out_hist_bin_edges = torch.linspace(self.out_min.item(), self.out_max.item(), steps=101)

        return nn.Identity()(x)

    
    
class ShiftDecorator:
    def __init__(self, shift):
        self.shift = shift
    
    def input_transform(self, x):
        self.shift = self.shift.to(x.device)
        return torch.sub(x, self.shift)
    
    def output_transform(self, x):
        return nn.Identity()(x)
    
class ScaleDecorator:
    def __init__(self, scale):
        self.scale = scale
    
    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        self.scale = self.scale.to(x.device)
        return self.scale * x
    
class ClampDecorator:
    def __init__(self, clamp_min, clamp_max):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    
    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        self.clamp_min = self.clamp_min.to(x.device)
        self.clamp_max = self.clamp_max.to(x.device)
        return DifferentiableClamp.apply(x, self.clamp_min, self.clamp_max)
    
class QuantizeDecorator:
    def __init__(self, levels_before_c, c):
        self.levels_before_c = levels_before_c
        self.c = c

    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        self.levels_before_c = self.levels_before_c.to(x.device)
        self.c = self.c.to(x.device)
        return StaircaseFunction.apply(x, self.levels_before_c / self.c)
    
class RandomMaskAdjustmentStrategy:
    def adjust(self, base, target, rate):
        random_mask = torch.rand(base.shape, device=base.device)
        random_mask = (random_mask < rate).float()
        return random_mask * target + (1.0 - random_mask) * base

class NestedAdjustmentStrategy:
    def __init__(self, strategies):
        self.strategies = strategies
    
    def adjust(self, base, target, rate):
        for strategy in self.strategies:
            target = strategy.adjust(base, target, rate)
        return target

class RateAdjustedDecorator:
    def __init__(self, rate, decorator, adjustment_strategy):
        self.rate = rate
        self.decorator = decorator
        self.adjustment_strategy = adjustment_strategy
    
    def input_transform(self, x):
        return self.adjustment_strategy.adjust(x, self.decorator.input_transform(x), self.rate)
    
    def output_transform(self, x):\
        return self.adjustment_strategy.adjust(x, self.decorator.output_transform(x), self.rate)


class NestedDecoration: 
    def __init__(self, decorators):
        self.decorators = decorators

    def input_transform(self, x):
        for decorator in reversed(self.decorators):
            x = decorator.input_transform(x)
        return x
    
    def output_transform(self, x):
        for decorator in self.decorators:
            x = decorator.output_transform(x)
        return x
    
class DecoratedActivation(nn.Module):
    def __init__(self, base_activation, decorator):
        super(DecoratedActivation, self).__init__()
        self.base_activation = base_activation
        self.decorator = decorator
    
    def forward(self, x):
        out = self.decorator.input_transform(x)
        out = self.base_activation(out)
        out = self.decorator.output_transform(out)
        return out

class TransformedActivation(nn.Module):
    def __init__(self, base_activation, decorators):
        super(TransformedActivation, self).__init__()
        self.base_activation = base_activation
        self.decorators = decorators
        self._update_activation()
    
    def decorate(self, decorator):
        self.decorators.append(decorator)
        self._update_activation()

    def pop_decorator(self):
        popped_decorator = self.decorators.pop()
        self._update_activation()
        return popped_decorator

    def forward(self, x):
        return self.act(x)
    
    def _update_activation(self):
        self.act = self.base_activation
        for decorator in self.decorators:
            self.act = DecoratedActivation(self.act, decorator)
        
    
class FrozenStatsNormalization(nn.Module):
    def __init__(self, normalization):
        super(FrozenStatsNormalization, self).__init__()
        self.running_mean = normalization.running_mean.clone().detach()
        self.running_var = normalization.running_var.clone().detach()
        self.weight = normalization.weight
        self.bias = normalization.bias
        self.eps = normalization.eps
    
        self.affine = normalization.affine
    
    def forward(self, x):
        self.weight = self.weight.to(x.device)
        self.bias = self.bias.to(x.device)
        self.running_mean = self.running_mean.to(x.device)
        self.running_var = self.running_var.to(x.device)
        
        # batch norm with frozen params
        return nn.functional.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias, False, 0, self.eps)
    

class MaxValueScaler(nn.Module):
    def __init__(self):
        super(MaxValueScaler, self).__init__()
        self.max_value = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    
    def forward(self, x):
        max_x = torch.max(x)
        
        if self.training:
            self.max_value.data = 0.1 * max_x + 0.9 * self.max_value

        return x / self.max_value
    
class AnyDecorator:
    def __init__(self, any_module):
        self.module = any_module
    
    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        return self.module(x)
    
class MixAdjustmentStrategy:
    def adjust(self, base, target, rate):
        return rate * target + (1.0 - rate) * base
    
class FrozenStatsMaxValueScaler(nn.Module):
    def __init__(self, scaler: MaxValueScaler):
        super(FrozenStatsMaxValueScaler, self).__init__()
        self.max_value = scaler.max_value.clone().detach()
    
    def forward(self, x):
        return x / self.max_value