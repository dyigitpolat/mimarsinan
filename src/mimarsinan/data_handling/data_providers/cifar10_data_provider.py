from mimarsinan.data_handling.data_provider import DataProvider

import torchvision.transforms as transforms
import torchvision

import torch

class CIFAR10_DataProvider(DataProvider):
    datasets_path = "../datasets"

    def __init__(self):
        super().__init__()

        train_transform = transforms.Compose([
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

        test_validation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

        base_training_dataset = torchvision.datasets.CIFAR10(
            root=self.datasets_path, train=True, download=True,
            transform=train_transform)
        
        training_validation_split = 0.99
        training_length = int(len(base_training_dataset) * training_validation_split)
        
        self.training_dataset = torch.utils.data.Subset(
            base_training_dataset, range(0, training_length))
        self.validation_dataset = torch.utils.data.Subset(
            base_training_dataset, range(training_length, len(base_training_dataset)))

        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.datasets_path, train=False, download=True,
            transform=test_validation_transform)

    def _get_training_dataset(self):
        return self.training_dataset

    def _get_validation_dataset(self):
        return self.validation_dataset

    def _get_test_dataset(self):
        return self.test_dataset