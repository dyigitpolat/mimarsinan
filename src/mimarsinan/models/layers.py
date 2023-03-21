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
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(WokeBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x):
        if len(x.shape) == 2:
            return nn.BatchNorm1d.forward(self, x)
        else:
            x = x.transpose(1, 2)
            x = nn.BatchNorm1d.forward(self, x)
            x = x.transpose(1, 2)
            return x
    
class SoftQuantize(nn.Module):
    def __init__(self, Tq):
        super(SoftQuantize, self).__init__()
        self.Tq = Tq
    
    def forward(self, x, alpha=4.5):
        h = 1.0 / self.Tq
        w = 1.0 / self.Tq
        a = torch.tensor(alpha)
        output = h * (
            0.5 * (1.0/torch.tanh(a/2)) * 
            torch.tanh(a * ((x/w-torch.floor(x/w))-0.5)) + 
            0.5 + torch.floor(x/w))
        return output

class LeakyClamp(nn.Module):
    def __init__(self, leak):
        super(LeakyClamp, self).__init__()
        self.leak = 1.0 - leak
    
    def forward(self, x):
        return nn.LeakyReLU()(
            torch.where(
                x < 0.0,
                self.leak * x,
                torch.where(
                    x < 1.0,
                    x,
                    self.leak * x)))

class CQ_Activation(nn.Module):
    def __init__(self, Tq):
        super(CQ_Activation, self).__init__()
        self.Tq = Tq
        self.soft_quantize = SoftQuantize(Tq)
    
    def forward(self, x):
        out = nn.ReLU()(x)
        out = torch.clamp(out, 0.0, 1.0)
        out = self.soft_quantize(out, 4.5)
        return out
    
class CQ_Activation_NoClamp(nn.Module):
    def __init__(self, Tq):
        super(CQ_Activation_NoClamp, self).__init__()
        self.Tq = Tq
        self.soft_quantize = SoftQuantize(Tq)
    
    def forward(self, x):
        out = nn.LeakyReLU()(x)
        out = self.soft_quantize(out, 4.5)
        return out
    
class CQ_Activation_LeakyClamp(nn.Module):
    def __init__(self, Tq, leak):
        super(CQ_Activation_LeakyClamp, self).__init__()
        self.Tq = Tq
        self.soft_quantize = SoftQuantize(Tq)
        self.leak = leak
    
    def forward(self, x):
        out = LeakyClamp(self.leak)(x)
        out = self.soft_quantize(out, 4.5)
        return out
    