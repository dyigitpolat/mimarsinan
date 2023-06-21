import torch.nn as nn
import torch

class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()

        self.eps = torch.tensor(1e-8)
        self.running_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.momentum = torch.tensor(0.1)

    def get_factor(self):
        return (1.0 / (self.running_mean + self.eps)) ** 0.5

    def forward(self, x):
        abs_mean = torch.mean(x**2)

        m = self.momentum
        self.running_mean = nn.Parameter(
            (1 - m)*self.running_mean + m*abs_mean, requires_grad=False)
        
        factor = self.get_factor().detach()
        return x * factor
    
class WokeBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, 
                 num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True):
        super(WokeBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x):
        if len(x.shape) == 2:
            return nn.BatchNorm1d.forward(self, x)
        else:
            x = x.transpose(1, 2)
            x = nn.BatchNorm1d.forward(self, x)
            x = x.transpose(1, 2)
            return x
    
    
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
        return grad_input, None

class SoftQuantize(nn.Module):
    def __init__(self, Tq):
        super(SoftQuantize, self).__init__()
        self.Tq = Tq
    
    def forward(self, x):
        return StaircaseFunction.apply(x, self.Tq)
    

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
    def forward(self, x):
        return DifferentiableClamp.apply(x, 0.0, 1.0)
    
class CQ_Activation(nn.Module):
    def __init__(self, Tq):
        super(CQ_Activation, self).__init__()
        self.Tq = Tq
        self.soft_quantize = SoftQuantize(Tq)
    
    def forward(self, x):
        out = ClampedReLU()(x)
        out = self.soft_quantize(out)
        return out
    
class CQ_Activation_Parametric(nn.Module):
    def __init__(self, Tq, rate, base_activation=None):
        super(CQ_Activation_Parametric, self).__init__()
        self.Tq = Tq
        self.rate = rate
        self.soft_quantize = SoftQuantize(Tq)
        if base_activation is None:
            base_activation = ShiftedActivation(ClampedReLU(), 0.5/Tq)
            
        self.base_activation = base_activation
    
    def forward(self, x):
        out_0 = self.base_activation(x)
        out_1 = ClampedReLU()(x)

        random_mask = torch.rand(x.shape, device=x.device)
        random_mask = (random_mask < self.rate).float()
        return \
            random_mask * self.soft_quantize(out_1) \
            + (1.0 - random_mask) * out_0
    
class SmoothStaircaseFunction(Function):
    @staticmethod
    def forward(ctx, x, Tq, alpha):
        h = 1.0 / Tq
        w = 1.0 / Tq
        a = torch.tensor(alpha)
        output = h * (
            0.5 * (1.0/torch.tanh(a/2)) * 
            torch.tanh(a * ((x/w-torch.floor(x/w))-0.5)) + 
            0.5 + torch.floor(x/w))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None
    
class SmoothStaircase(nn.Module):
    def __init__(self, Tq, alpha):
        super(SmoothStaircase, self).__init__()
        self.Tq = Tq
        self.alpha = alpha
    
    def forward(self, x):
        return SmoothStaircaseFunction.apply(x, self.Tq, self.alpha)

class CQ_Activation_Soft(nn.Module):
    def __init__(self, Tq, alpha):
        super(CQ_Activation_Soft, self).__init__()
        self.Tq = Tq
        self.alpha = alpha
        self.staircase = SmoothStaircase(Tq, alpha)
    
    def forward(self, x):
        out = x - 0.5/self.Tq
        out = self.staircase(out)
        out = ClampedReLU()(out)
        return out
    
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