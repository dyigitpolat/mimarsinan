from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

import torchvision.transforms as transforms
import torchvision

import torch


@BasicDataProviderFactory.register("MNIST32_DataProvider")
class MNIST32_DataProvider(DataProvider):
    """
    MNIST resized to 32x32 (still 1 channel).
    This is required for VGG-style architectures with 5x (2x2,stride2) pooling:
      32 -> 16 -> 8 -> 4 -> 2 -> 1
    """

    def __init__(self, datasets_path, *, seed: int | None = 0):
        super().__init__(datasets_path, seed=seed)

        tfm = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

        base_training_dataset = torchvision.datasets.MNIST(
            root=self.datasets_path, train=True, download=True, transform=tfm
        )

        training_validation_split = 0.99
        base_training_length = len(base_training_dataset)
        training_length = int(base_training_length * training_validation_split)
        validation_length = base_training_length - training_length

        self.training_dataset, self.validation_dataset = torch.utils.data.random_split(
            base_training_dataset,
            (training_length, validation_length),
            generator=self._get_split_generator(),
        )

        self.test_dataset = torchvision.datasets.MNIST(
            root=self.datasets_path, train=False, download=True, transform=tfm
        )

    def _get_training_dataset(self):
        return self.training_dataset

    def _get_validation_dataset(self):
        return self.validation_dataset

    def _get_test_dataset(self):
        return self.test_dataset

    def get_prediction_mode(self):
        return ClassificationMode(10)


