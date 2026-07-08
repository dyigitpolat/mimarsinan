from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
from mimarsinan.data_handling.preprocessing import register_normalization_preset

import torchvision.transforms as transforms
import torchvision

import torch
import os

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
register_normalization_preset("cifar100", CIFAR100_MEAN, CIFAR100_STD)


@BasicDataProviderFactory.register("CIFAR100_DataProvider")
class CIFAR100_DataProvider(DataProvider):
    DISPLAY_LABEL = "CIFAR-100 (32×32×3, 100 classes)"

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        super().__init__(datasets_path, seed=seed, preprocessing=preprocessing, batch_size=batch_size)

        path_str = str(self.datasets_path + '/cifar-100-python')
        download = not os.path.exists(path_str)

        full_train = torchvision.datasets.CIFAR100(
            root=self.datasets_path, train=True, download=download, transform=None,
        )
        cut = int(len(full_train) * 0.95)
        self._train_raw = torch.utils.data.Subset(full_train, range(0, cut))
        self._val_raw   = torch.utils.data.Subset(full_train, range(cut, len(full_train)))
        self._test_raw  = torchvision.datasets.CIFAR100(
            root=self.datasets_path, train=False, download=download, transform=None,
        )

    def get_prediction_mode(self):
        return ClassificationMode(100)

    def raw_datasets(self) -> dict:
        return {"train": self._train_raw, "val": self._val_raw, "test": self._test_raw}

    def torch_transforms(self) -> dict:
        return {
            "train": [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
            ],
            "val":  [transforms.ToTensor()],
            "test": [transforms.ToTensor()],
        }

    def ffcv_transforms(self) -> dict:
        return {
            "splits": {
                "train": [
                    ("SimpleRGBImageDecoder", {}),
                    ("RandomTranslate", {"padding": 4}),
                    ("Cutout", {"crop_size": 16}),
                    ("RandomBrightness", {"magnitude": 0.3}),
                    ("RandomContrast", {"magnitude": 0.3}),
                    ("RandomSaturation", {"magnitude": 0.3}),
                ],
                "val":  [("SimpleRGBImageDecoder", {})],
                "test": [("SimpleRGBImageDecoder", {})],
            },
        }
