from mimarsinan.models.layers import *

import torch.nn as nn
import torch

class SimpleMLP(nn.Module):
    def __init__(
        self, 
        inner_mlp_width,
        inner_mlp_count,
        input_size,
        output_size,
        bias=True):
        super(SimpleMLP, self).__init__()

        self.layers = \
            nn.ModuleList([
                nn.Linear(
                    in_features=input_size, out_features=inner_mlp_width,
                    bias=bias),
            ])
        self.bns = []

        for _ in range(inner_mlp_count):
            self.layers.append(
                nn.Linear(
                    in_features=inner_mlp_width, out_features=inner_mlp_width,
                    bias=bias))
            self.bns.append(WokeBatchNorm1d(inner_mlp_width))

        self.layers.append(
            nn.Linear(
                in_features=inner_mlp_width, out_features=output_size,
                bias=bias))
        

    def forward(self, x):
        output = x.view(x.size(0), -1)
        i = 0
        for layer in self.layers:
            output = layer(output)
            if i > 1 and i < len(self.layers) - 1:
                output = self.bns[i - 1](output)
            output = nn.LeakyReLU()(output)
            i += 1

        return output

class SoftQuantize(nn.Module):
    def __init__(self, Tq):
        super(SoftQuantize, self).__init__()
        self.Tq = Tq
    
    def forward(self, x, alpha=10.0):
        h = 1.0 / self.Tq
        w = 1.0 / self.Tq
        a = torch.tensor(alpha)
        output = h * (
            0.5 * (1.0/torch.tanh(a/2)) * 
            torch.tanh(a * ((x/w-torch.floor(x/w))-0.5)) + 
            0.5 + torch.floor(x/w))
        return output

class SimpleMLP_CQ(nn.Module):
    def __init__(self, model, Tq):
        super(SimpleMLP_CQ, self).__init__()
        self.model = model
        self.layers = nn.ModuleList(model.layers)
        self.Tq = Tq
        self.alpha = 4.5
    
    def forward(self, x):
        output = x.view(x.size(0), -1)

        # soft quantize the input
        output = SoftQuantize(self.Tq)(output)

        for layer in self.layers:
            output = layer(output)
            output = nn.LeakyReLU()(output)

            # clamp 
            output = torch.clamp(output, 0.0, 1.0)

            # soft quantize
            output = SoftQuantize(self.Tq)(output, self.alpha)

        return output


