from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator

import torch.nn as nn
import torch

class InputCQ(nn.Module):
    def __init__(self, Tq):
        super(InputCQ, self).__init__()
        self.in_act = TransformedActivation(
            base_activation = nn.Identity(),
            decorators = [
                ClampDecorator(torch.tensor(0.0), torch.tensor(1.0)),
                QuantizeDecorator(torch.tensor(Tq), torch.tensor(1.0))
            ])

    def forward(self, x):
        x = self.in_act(x)
        return x