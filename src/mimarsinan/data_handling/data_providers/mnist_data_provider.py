from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
import torchvision.transforms as transforms
import torchvision

import torch

@BasicDataProviderFactory.register("MNIST_DataProvider")
class MNIST_DataProvider(DataProvider):
    DISPLAY_LABEL = "MNIST (28×28, 10 classes)"

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        super().__init__(datasets_path, seed=seed, preprocessing=preprocessing, batch_size=batch_size)

        train_transform = self._apply_preprocessing([transforms.ToTensor()], train=True)
        eval_transform = self._apply_preprocessing([transforms.ToTensor()], train=False)

        base_training_dataset = torchvision.datasets.MNIST(
            root=self.datasets_path, train=True, download=True,
            transform=train_transform)

        training_validation_split = 0.95
        
        base_training_length = len(base_training_dataset)
        training_length = int(base_training_length * training_validation_split)
        validation_length = base_training_length - training_length

        self.training_dataset, self.validation_dataset = torch.utils.data.random_split(
            base_training_dataset,
            (training_length, validation_length),
            generator=self._get_split_generator(),
        )

        self.test_dataset = torchvision.datasets.MNIST(
            root=self.datasets_path, train=False, download=True,
            transform=eval_transform)

    def _get_training_dataset(self):
        return self.training_dataset

    def _get_validation_dataset(self):
        return self.validation_dataset

    def _get_test_dataset(self):
        return self.test_dataset
    
    def get_prediction_mode(self):
        return ClassificationMode(10)
