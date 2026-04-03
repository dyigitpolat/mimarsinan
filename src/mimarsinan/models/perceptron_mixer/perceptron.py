from mimarsinan.models.layers import MaxValueScaler, LeakyGradReLU


import torch.nn as nn
import torch


# Canonical mapping from string names to activation constructors.
# "ReLU" maps to LeakyGradReLU (ReLU forward, leaky gradient backward) —
# the default pipeline activation for SNN training.
ACTIVATION_REGISTRY = {
    "ReLU": LeakyGradReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
}


def make_activation(name=None):
    """Create an activation module from a string name. Default: LeakyGradReLU."""
    if name is None or name not in ACTIVATION_REGISTRY:
        return LeakyGradReLU()
    return ACTIVATION_REGISTRY[name]()


class Perceptron(nn.Module):
    def __init__(
        self,
        output_channels, input_features, bias=True,
        normalization=nn.Identity(),
        base_activation_name=None,
        name="Perceptron"):

        super(Perceptron, self).__init__()
        self.name = name

        self.input_features = input_features
        self.output_channels = output_channels

        self.layer = nn.Linear(
            input_features, output_channels, bias=bias)

        self.normalization = normalization

        self.scaler = nn.Identity()
        self.base_scaler = MaxValueScaler()
        self.scale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.input_activation = nn.Identity()

        self.base_activation_name = base_activation_name or "ReLU"
        self.base_activation = make_activation(self.base_activation_name)
        self.activation = make_activation(self.base_activation_name)

        self.regularization = nn.Identity()

        self.parameter_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.input_activation_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.activation_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.per_input_scales = None

        # True for the first perceptron of each neural segment (host-side ComputeOp in IR).
        self.is_encoding_layer = False

    def set_parameter_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.parameter_scale.data = new_scale.data
    
    def set_activation_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.activation_scale.data = new_scale.data

    def set_input_activation_scale(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.input_activation_scale.data = new_scale.data

    def set_scale_factor(self, new_scale):
        if isinstance(new_scale, float):
            new_scale = torch.tensor(new_scale)
        self.scale_factor.data = new_scale.data
        
    def set_activation(self, activation):
        self.activation = activation

    def set_regularization(self, regularizer):
        self.regularization = regularizer

    def set_scaler(self, scaler):
        self.scaler = scaler

    def forward(self, x):
        x = self.input_activation(x)
        
        out = self.layer(x)
        out = self.normalization(out)
        out = self.scaler(out)

        out = self.activation(out)
        
        if self.training:
            out = self.regularization(out)
        
        return out
