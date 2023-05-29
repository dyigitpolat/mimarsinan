from mimarsinan.data_handling.data_provider import DataProvider


import torchvision.transforms as transforms
import torchvision

class MNIST_DataProvider(DataProvider):
    datasets_path = "../datasets"

    def __init__(self):
        super().__init__()
        self.datasets = {}

    def _get_training_dataset(self):
        if "training" not in self.datasets:
            self.datasets["training"] = torchvision.datasets.MNIST(
            root=self.datasets_path, train=True, download=True,
            transform=transforms.ToTensor())

        return self.datasets["training"]

    def _get_validation_dataset(self):
        if "validation" not in self.datasets:
            self.datasets["validation"] = torchvision.datasets.MNIST(
            root=self.datasets_path, train=True, download=True,
            transform=transforms.ToTensor())

        return self.datasets["validation"]

    def _get_test_dataset(self):
        if "test" not in self.datasets:
            self.datasets["test"] = torchvision.datasets.MNIST(
            root=self.datasets_path, train=False, download=True,
            transform=transforms.ToTensor())

        return self.datasets["test"]
