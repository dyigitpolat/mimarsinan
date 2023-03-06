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


        

