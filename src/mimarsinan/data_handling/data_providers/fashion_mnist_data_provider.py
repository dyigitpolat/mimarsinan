from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
import torchvision.transforms as transforms
import torchvision

import torch


@BasicDataProviderFactory.register("FashionMNIST_DataProvider")
class FashionMNIST_DataProvider(DataProvider):
    DISPLAY_LABEL = "Fashion-MNIST (28×28, 10 classes)"

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        super().__init__(datasets_path, seed=seed, preprocessing=preprocessing, batch_size=batch_size)

        full_train = torchvision.datasets.FashionMNIST(
            root=self.datasets_path, train=True, download=True, transform=None,
        )
        n = len(full_train)
        train_n = int(n * 0.95)
        val_n = n - train_n
        self._train_raw, self._val_raw = torch.utils.data.random_split(
            full_train, (train_n, val_n), generator=self._get_split_generator(),
        )
        self._test_raw = torchvision.datasets.FashionMNIST(
            root=self.datasets_path, train=False, download=True, transform=None,
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

    # No FFCV opt-in: FFCV's RGBImageField requires 3 channels and there is
    # no stock op to collapse back to 1.
