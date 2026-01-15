from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

import torchvision.transforms as transforms
import torchvision

import torch
import os

@BasicDataProviderFactory.register("CIFAR100_DataProvider")
class CIFAR100_DataProvider(DataProvider):
    def __init__(self, datasets_path, *, seed: int | None = 0):
        super().__init__(datasets_path, seed=seed)

        path_str = str(self.datasets_path + '/cifar-100-python')
        download = not os.path.exists(path_str)

        train_transform = transforms.Compose([
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor()
            #transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

        test_validation_transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

        training_dataset = torchvision.datasets.CIFAR100(
            root=self.datasets_path, train=True, download=download,
            transform=train_transform)
        
        validation_dataset = torchvision.datasets.CIFAR100(
            root=self.datasets_path, train=True, download=download,
            transform=test_validation_transform)
        
        training_validation_split = 0.99
        training_length = int(len(training_dataset) * training_validation_split)
        
        self.training_dataset = torch.utils.data.Subset(
            training_dataset, range(0, training_length))
        self.validation_dataset = torch.utils.data.Subset(
            validation_dataset, range(training_length, len(training_dataset)))

        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.datasets_path, train=False, download=download,
            transform=test_validation_transform)

    def _get_training_dataset(self):
        return self.training_dataset

    def _get_validation_dataset(self):
        return self.validation_dataset

    def _get_test_dataset(self):
        return self.test_dataset

    def get_prediction_mode(self):
        return ClassificationMode(100)
