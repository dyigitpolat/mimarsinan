from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator, NoiseDecorator

import torch.nn as nn
import torch
class Supermodel(nn.Module):
    def __init__(self, device, input_shape, num_classes, preprocessor, perceptron_flow, Tq):
        super(Supermodel, self).__init__()
        self.device = device
        self.preprocessor = preprocessor
        self.perceptron_flow = perceptron_flow

        self.in_act = TransformedActivation(
            base_activation = nn.Identity(),
            decorators = [
                ClampDecorator(torch.tensor(0.0), torch.tensor(1.0)),
                QuantizeDecorator(torch.tensor(Tq), torch.tensor(1.0))
            ])

    def forward(self, x):
        out = self.preprocessor(x)
        out = self.in_act(out)
        out = self.perceptron_flow(out)
        return out
    
    def get_preprocessor(self):
        return self.preprocessor
        
    def get_preprocessor_output_shape(self):
        return self.preprocessor(torch.zeros(1, *self.input_shape)).shape[1:]
    
    def get_perceptrons(self):
        return self.perceptron_flow.get_perceptrons()
    
    def get_mapper_repr(self):
        return self.perceptron_flow.get_mapper_repr()
    
    def get_input_activation(self):
        return self.perceptron_flow.get_input_activation()
    
    def set_input_activation(self, activation):
        self.perceptron_flow.set_input_activation(activation)