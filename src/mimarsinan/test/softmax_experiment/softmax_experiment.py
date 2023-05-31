from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_providers.mnist_data_provider import MNIST_DataProvider
from mimarsinan.model_training.training_utilities import BasicClassificationLoss

import torch.nn as nn
import torch

class SoftmaxWithInputStats(nn.Module):
    def __init__(self):
        super(SoftmaxWithInputStats, self).__init__()
        self.mean_of_maximums = None
        self.variance_of_maximums = None

    def update_stats(self, x):
        in_ = x.reshape(x.shape[0], -1)
        argmax = torch.argmax(in_, dim=1)
        max_values = torch.gather(in_, 1, argmax.unsqueeze(1)).squeeze(1)
        
        curr_mean_of_maximums = torch.mean(max_values, dim=0)
        curr_variance_of_maximums = torch.var(max_values, dim=0)
        if self.mean_of_maximums is None:
            self.mean_of_maximums = curr_mean_of_maximums
            self.variance_of_maximums = curr_variance_of_maximums
        else:
            self.mean_of_maximums = \
                0.9 * self.mean_of_maximums + 0.1 * curr_mean_of_maximums
            self.variance_of_maximums = \
                0.9 * self.variance_of_maximums + 0.1 * curr_variance_of_maximums

    def forward(self, x):
        self.update_stats(x)
        return nn.functional.softmax(x, dim=1)
    
class FusedSoftmax(nn.Module):
    def __init__(self, mean, variance):
        super(FusedSoftmax, self).__init__()
        self.mean = mean
        self.threshold = mean - 4 * variance.sqrt()
        print("Fused softmax threshold:", self.threshold)

    def forward(self, x):
        return torch.where(
            x > self.threshold,
            x * (1.0 / self.threshold),
            torch.zeros_like(x)
        )

def run_softmax_experiment():
    print("setting up...")
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 256),
        nn.LeakyReLU(),
        SoftmaxWithInputStats(),
        nn.Linear(256, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(),
        nn.Linear(256, 10)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_provider = MNIST_DataProvider()
    trainer = BasicTrainer(model, device, data_provider, BasicClassificationLoss())

    print("training...")
    trainer.train_n_epochs(0.001, 10)

    print("testing...")
    print("acc:", trainer.test())

    print("softmax layer mean of maximums:", model[5].mean_of_maximums)
    print("softmax layer mean of maximums:", model[5].variance_of_maximums)

    print("fusing softmax...")
    model[5] = FusedSoftmax(model[5].mean_of_maximums, model[5].variance_of_maximums)

    print("testing...")
    print("acc:", trainer.test())

    print("fine tuning for one epoch...")
    trainer.train_n_epochs(0.001, 1)

    print("testing...")
    print("acc after fine tune:", trainer.test())
