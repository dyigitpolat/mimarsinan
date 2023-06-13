import torch.nn as nn

class PerceptronFlow(nn.Module):
    def __init__(self):
        super(PerceptronFlow, self).__init__()
        self.activation = nn.LeakyReLU()

    def fuse_normalization(self):
        for layer in self.get_perceptrons():
            layer.fuse_normalization()

    def set_activation(self, activation):
        self.activation = activation
        for layer in self.get_perceptrons():
            layer.set_activation(activation)

    def set_regularization(self, regularizer):
        for layer in self.get_perceptrons():
            layer.set_regularization(regularizer)
    
    def get_perceptrons(self):
        raise NotImplementedError
    
    def get_mapper_repr(self):
        raise NotImplementedError