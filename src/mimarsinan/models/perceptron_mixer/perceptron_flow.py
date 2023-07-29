import torch.nn as nn

class PerceptronFlow(nn.Module):
    def __init__(self):
        super(PerceptronFlow, self).__init__()
    
    def get_perceptrons(self):
        raise NotImplementedError
    
    def get_mapper_repr(self):
        raise NotImplementedError