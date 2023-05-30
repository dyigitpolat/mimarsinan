from mimarsinan.data_handling.data_provider import DataProvider

import torchvision.transforms as transforms
import torchvision

import torch

class MNIST_DataProvider(DataProvider):
    datasets_path = "../datasets"

    def __init__(self):
        super().__init__()

        base_training_dataset = torchvision.datasets.MNIST(
            root=self.datasets_path, train=True, download=True,
            transform=transforms.ToTensor())

        training_validation_split = 0.99
        
        base_training_length = len(base_training_dataset)
        training_length = int(base_training_length * training_validation_split)
        validation_length = base_training_length - training_length

        self.training_dataset, self.validation_dataset = \
            torch.utils.data.random_split( 
                base_training_dataset, (training_length, validation_length))

        self.test_dataset = torchvision.datasets.MNIST(
            root=self.datasets_path, train=False, download=True,
            transform=transforms.ToTensor())

    def _get_training_dataset(self):
        return self.training_dataset

    def _get_validation_dataset(self):
        return self.validation_dataset

    def _get_test_dataset(self):
        return self.test_dataset
