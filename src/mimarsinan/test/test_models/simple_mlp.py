import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(
        self, 
        inner_mlp_width,
        inner_mlp_count,
        input_size,
        output_size):
        super(SimpleMLP, self).__init__()

        self.layers = \
            nn.ModuleList([
                nn.Linear(in_features=input_size, out_features=inner_mlp_width),
            ])

        for _ in range(inner_mlp_count):
            self.layers.append(
                nn.Linear(
                    in_features=inner_mlp_width, out_features=inner_mlp_width))

        self.layers.append(
            nn.Linear(in_features=inner_mlp_width, out_features=output_size))

    def forward(self, x):
        output = x.view(x.size(0), -1)
        for layer in self.layers:
            output = layer(output)
            output = nn.ReLU()(output)

        return output
