"""Fused linear layer used before soft-core mapping unfuses bias."""

import torch
import torch.nn as nn


class FusedLinear(nn.Module):
    """Linear layer with bias fused into an extra weight column."""

    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features + 1, output_features, bias=False)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        bias_feature = torch.ones((batch_size, seq_len, 1), device=x.device)
        x = torch.cat([x, bias_feature], dim=-1)
        output = self.linear(x)

        if len(output.shape) == 3 and output.shape[1] == 1:
            output = output.squeeze(1)
        return output
