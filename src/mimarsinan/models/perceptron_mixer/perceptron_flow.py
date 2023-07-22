import torch.nn as nn

class PerceptronFlow(nn.Module):
    def __init__(self):
        super(PerceptronFlow, self).__init__()

    def fuse_normalization(self):
        for layer in self.get_perceptrons():
            layer.fuse_normalization()
    
    def get_perceptrons(self):
        raise NotImplementedError
    
    def get_mapper_repr(self):
        raise NotImplementedError