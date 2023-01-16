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

        for _ in range(inner_mlp_count):
            self.layers.append(
                nn.Linear(
                    in_features=inner_mlp_width, out_features=inner_mlp_width,
                    bias=bias))

        self.layers.append(
            nn.Linear(
                in_features=inner_mlp_width, out_features=output_size,
                bias=bias))
        

    def forward(self, x):
        output = x.view(x.size(0), -1)
        for layer in self.layers:
            output = layer(output)
            output = nn.LeakyReLU()(output)

        return output

class SimpleMLP_CQ(nn.Module):
    def __init__(self, model, Tq):
        super(SimpleMLP_CQ, self).__init__()
        self.model = model
        self.Tq = Tq
    
    def forward(self, x):
        output = x.view(x.size(0), -1)
        for layer in self.model.layers:
            output = layer(output)
            output = nn.LeakyReLU()(output)

            # quantize
            output = torch.round(output * self.Tq) / self.Tq

        return output
