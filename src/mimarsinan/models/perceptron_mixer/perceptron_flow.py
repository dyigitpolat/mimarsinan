import torch.nn as nn

class PerceptronFlow(nn.Module):
    def __init__(self, device):
        super(PerceptronFlow, self).__init__()
        self.device = device
    
    def get_perceptrons(self):
        raise NotImplementedError
    
    def get_mapper_repr(self):
        raise NotImplementedError