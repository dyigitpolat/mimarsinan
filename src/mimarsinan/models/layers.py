import torch.nn as nn
import torch
    
from torch.autograd import Function

class StaircaseFunction(Function):
    @staticmethod
    def forward(ctx, x, Tq, scale = 1.0):
        #Tq = torch.tensor(Tq)
        #ctx.save_for_backward(x, Tq)
        return torch.floor(x * (Tq/scale)) / (Tq/scale)

    @staticmethod
    def backward(ctx, grad_output):
        #x, Tq = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None

class SoftQuantize(nn.Module):
    def __init__(self, Tq, scale = 1.0):
        super(SoftQuantize, self).__init__()
        self.Tq = Tq
        self.scale = scale
    
    def forward(self, x):
        return StaircaseFunction.apply(x, self.Tq, self.scale)
    

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
    
class ClampedReLU(nn.Module):
    def __init__(self, clamp_min=0.0, clamp_max=1.0):
        super(ClampedReLU, self).__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        return DifferentiableClamp.apply(x, self.clamp_min, self.clamp_max)
    
class ClampedReLU_Parametric(nn.Module):
    def __init__(self, rate, base_activation):
        super(ClampedReLU_Parametric, self).__init__()
        self.rate = rate
        self.base_activation = base_activation

    def forward(self, x):
        base_act = self.base_activation(x)
        base_max = torch.max(base_act).item()
        base_min = torch.min(base_act).item()
        
        clamp_max = base_max
        clamp_min = (1.0 - self.rate) * base_min + self.rate * 0.0

        random_mask = torch.rand(x.shape, device=x.device)
        random_mask = (random_mask < self.rate).float()

        return \
            random_mask * ClampedReLU(clamp_min, clamp_max)(x) \
            + (1.0 - random_mask) * base_act
    
class CQ_Activation(nn.Module):
    def __init__(self, Tq, clamp_max=1.0):
        super(CQ_Activation, self).__init__()
        self.Tq = Tq
        self.clamp_max = clamp_max
    
    def forward(self, x):
        out = ClampedReLU(clamp_min=0.0, clamp_max=self.clamp_max)(x)
        out = SoftQuantize(self.Tq, scale=self.clamp_max)(out)
        return out
    
class CQ_Activation_Parametric(nn.Module):
    def __init__(self, Tq, rate, base_activation=None, clamp_max=1.0):
        super(CQ_Activation_Parametric, self).__init__()
        self.Tq = Tq
        self.rate = rate
        self.soft_quantize = SoftQuantize(Tq, scale=clamp_max)
        self.clamp_max = clamp_max

        if base_activation is None:
            base_activation = ShiftedActivation(
                ClampedReLU(clamp_min=0.0, clamp_max=clamp_max), 0.5/Tq)
            
        self.base_activation = base_activation
    
    def forward(self, x):
        out_0 = self.base_activation(x)
        out_1 = ClampedReLU(clamp_min=0.0, clamp_max=self.clamp_max)(x)

        random_mask = torch.rand(x.shape, device=x.device)
        random_mask = (random_mask < self.rate).float()
        return \
            random_mask * self.soft_quantize(out_1) \
            + (1.0 - random_mask) * out_0
    
class ShiftedActivation(nn.Module):
    def __init__(self, activation, shift):
        super(ShiftedActivation, self).__init__()
        self.activation = activation
        self.shift = shift
    
    def forward(self, x):
        return self.activation(x - self.shift)
    
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
    
class ScaleActivation(nn.Module):
    def __init__(self, base_activation, scale):
        super(ScaleActivation, self).__init__()
        self.base_activation = base_activation
        self.scale = scale
    
    def forward(self, x):
        return self.scale * self.base_activation(x)
    
class ParametricScaleActivation(nn.Module):
    def __init__(self, base_activation, scale, rate):
        super(ParametricScaleActivation, self).__init__()
        self.base_activation = base_activation
        self.scale = scale
        self.rate = rate
    
    def forward(self, x):
        out_0 = self.base_activation(x)
        out_1 = self.scale * self.base_activation(x)

        return \
            self.rate * out_1 + (1.0 - self.rate) * out_0

