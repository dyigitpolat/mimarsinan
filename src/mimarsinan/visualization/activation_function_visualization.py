import matplotlib.pyplot as plt

import torch

class ActivationFunctionVisualizer:
    def __init__(self, activation_function):
        self.activation_function = activation_function
        self.x = torch.arange(-1.5, 1.5, 0.001)
        self.y = torch.tensor([self.activation_function(x) for x in self.x])

    def plot(self, filename):
        plt.grid()
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.plot(self.x, self.y)
        plt.savefig(filename)
        plt.clf()
