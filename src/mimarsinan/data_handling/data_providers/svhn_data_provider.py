from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

import torchvision.transforms as transforms
import torchvision

import torch


@BasicDataProviderFactory.register("SVHN_DataProvider")
class SVHN_DataProvider(DataProvider):
    DISPLAY_LABEL = "SVHN (32×32×3, 10 classes)"

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        super().__init__(datasets_path, seed=seed, preprocessing=preprocessing, batch_size=batch_size)

        # torchvision already remaps SVHN's MAT label 10 -> class 0, so this is a stock 10-class problem.
        full_train = torchvision.datasets.SVHN(
            root=self.datasets_path, split="train", download=True, transform=None,
        )
        n = len(full_train)
        train_n = int(n * 0.95)
        val_n = n - train_n
        self._train_raw, self._val_raw = torch.utils.data.random_split(
            full_train, (train_n, val_n), generator=self._get_split_generator(),
        )
        self._test_raw = torchvision.datasets.SVHN(
            root=self.datasets_path, split="test", download=True, transform=None,
        )

    def get_prediction_mode(self):
        return ClassificationMode(10)

    def raw_datasets(self) -> dict:
        return {"train": self._train_raw, "val": self._val_raw, "test": self._test_raw}

    def torch_transforms(self) -> dict:
        return {
            "train": [transforms.ToTensor()],
            "val":   [transforms.ToTensor()],
            "test":  [transforms.ToTensor()],
        }
