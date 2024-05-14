from mimarsinan.models.layers import MaxValueScaler


import torch.nn as nn
import torch

class Perceptron(nn.Module):
    def __init__(
        self, 
        output_channels, input_features, bias=True,
        normalization = nn.Identity(), name="Perceptron"):

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
        self.activation = nn.LeakyReLU()

        self.regularization = nn.Identity()
        
        self.parameter_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.input_activation_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.activation_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.base_activation = nn.LeakyReLU()

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
