from mimarsinan.mapping.mapping_utils import *

import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(
        self, 
        output_channels, input_features, bias=True,
        normalization = nn.Identity()):

        super(Perceptron, self).__init__()
        
        self.input_features = input_features
        self.output_channels = output_channels

        self.layer = nn.Linear(
            input_features, output_channels, bias=bias)

        self.normalization = normalization
        self.activation = nn.LeakyReLU()

        self.regularization = nn.Identity()
        self.base_threshold = 1.0
    
    def set_activation(self, activation):
        self.activation = activation

    def fuse_normalization(self):
        if isinstance(self.normalization, nn.Identity):
            return

        assert isinstance(self.normalization, nn.BatchNorm1d)
        assert self.normalization.affine

        w, b = get_fused_weights(
            linear_layer=self.layer, bn_layer=self.normalization)

        self.layer = nn.Linear(
            self.input_features, self.output_channels, bias=True)
        self.layer.weight.data = w
        self.layer.bias.data = b

        self.normalization = nn.Identity()

    def set_regularization(self, regularizer):
        self.regularization = regularizer

    def forward(self, x):
        out = self.layer(x)
        out = self.normalization(out)
        out = self.activation(out)
        
        if self.training:
            out = self.regularization(out)
        
        return out