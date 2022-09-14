import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, inner_mlp = 3):
        super(SimpleMLP, self).__init__()
        self.layers = \
            nn.ModuleList([
                nn.Linear(in_features=2, out_features=inner_mlp),
                nn.Linear(in_features=inner_mlp, out_features=inner_mlp),
                nn.Linear(in_features=inner_mlp, out_features=2) 
            ])

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
            output = nn.ReLU()(output)

        return output
