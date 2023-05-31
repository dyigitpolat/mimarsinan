from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_providers.mnist_data_provider import MNIST_DataProvider
from mimarsinan.model_training.training_utilities import BasicClassificationLoss

import torch.nn as nn
import torch
    
def run_softmax_experiment():
    print("setting up...")
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 256),
        nn.LeakyReLU(),
        nn.Softmax(),
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

    print("fusing softmax...")
    model[5] = nn.Identity()

    print("testing...")
    print("acc:", trainer.test())

    print("fine tuning for one epoch...")
    trainer.train_n_epochs(0.001, 1)

    print("testing...")
    print("acc after fine tune:", trainer.test())
