import torch.nn as nn
import torch

class Supermodel(nn.Module):
    def __init__(self, device, input_shape, num_classes, preprocessor, perceptron_flow):
        super(Supermodel, self).__init__()
        self.device = device
        self.preprocessor = preprocessor
        self.perceptron_flow = perceptron_flow

    def forward(self, x):
        out = self.preprocessor(x)
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