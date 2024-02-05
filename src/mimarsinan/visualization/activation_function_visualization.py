import matplotlib.pyplot as plt

import torch

class ActivationFunctionVisualizer:
    def __init__(self, activation_function, min_x = -1.5, max_x = 1.5, step = 0.001, device = 'cpu'):
        self.activation_function = activation_function
        self.max_x = max_x
        self.min_x = min_x

        self.x = torch.arange(min_x, max_x, step, device=device)
        self.y = self.activation_function(self.x)

    def plot(self, filename):
        plt.grid()
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.ylim(self.min_x / 2, self.max_x)
        plt.plot(self.x.cpu(), self.y.cpu())
        plt.savefig(filename)
        plt.clf()

        plt.close()
