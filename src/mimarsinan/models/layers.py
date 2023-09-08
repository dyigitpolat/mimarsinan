import torch.nn as nn
import torch
    
from torch.autograd import Function

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
        a = torch.tensor(a, device=x.device)
        b = torch.tensor(b, device=x.device)
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
    
class ActivationStats(nn.Module):
    def __init__(self, base_activation):
        super(ActivationStats, self).__init__()
        self.base_activation = base_activation
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('var', torch.zeros(1))
        self.register_buffer('max', torch.zeros(1))
        self.register_buffer('min', torch.zeros(1))
    
    def forward(self, x):
        out = self.base_activation(x)
        self.mean = torch.mean(out)
        self.var = torch.var(out)
        self.max = torch.max(out)
        self.min = torch.min(out)
        return out
    
    
class ShiftDecorator:
    def __init__(self, shift):
        self.shift = shift
    
    def input_transform(self, x):
        return x - self.shift
    
    def output_transform(self, x):
        return nn.Identity()(x)
    
class ScaleDecorator:
    def __init__(self, scale):
        self.scale = scale
    
    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        return self.scale * x
    
class ClampDecorator:
    def __init__(self, clamp_min, clamp_max):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    
    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        return DifferentiableClamp.apply(x, self.clamp_min, self.clamp_max)
    
class QuantizeDecorator:
    def __init__(self, Tq):
        self.Tq = Tq

    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        return StaircaseFunction.apply(x, self.Tq)
    
class RandomMaskAdjustmentStrategy:
    def adjust(self, base, target, rate):
        random_mask = torch.rand(base.shape, device=base.device)
        random_mask = (random_mask < rate).float()
        return random_mask * target + (1.0 - random_mask) * base

class RateAdjustedDecorator:
    def __init__(self, rate, decorator, adjustment_strategy):
        self.rate = rate
        self.decorator = decorator
        self.adjustment_strategy = adjustment_strategy
    
    def input_transform(self, x):
        return self.adjustment_strategy.adjust(x, self.decorator.input_transform(x), self.rate)
    
    def output_transform(self, x):\
        return self.adjustment_strategy.adjust(x, self.decorator.output_transform(x), self.rate)
    
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
        self.act = base_activation
        for decorator in decorators:
            self.act = DecoratedActivation(self.act, decorator)

    def forward(self, x):
        return self.act(x)
    
class CQ_Activation(nn.Module):
    def __init__(self, Tq, clamp_max=1.0):
        super(CQ_Activation, self).__init__()
        self.Tq = Tq
        self.clamp_max = clamp_max
    
    def forward(self, x):
        out = TransformedActivation(nn.ReLU(), 
            [ClampDecorator(0.0, self.clamp_max), QuantizeDecorator(self.Tq)])(x)
        return out