from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

import torchvision.transforms as transforms
import torchvision

import torch
import os


@BasicDataProviderFactory.register("CIFAR10_DataProvider")
class CIFAR10_DataProvider(DataProvider):
    DISPLAY_LABEL = "CIFAR-10 (32×32×3, 10 classes)"

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        super().__init__(datasets_path, seed=seed, preprocessing=preprocessing, batch_size=batch_size)

        path_str = str(self.datasets_path + '/cifar-10-batches-py')
        download = not os.path.exists(path_str)

        full_train = torchvision.datasets.CIFAR10(
            root=self.datasets_path, train=True, download=download, transform=None,
        )
        cut = int(len(full_train) * 0.95)
        self._train_raw = torch.utils.data.Subset(full_train, range(0, cut))
        self._val_raw   = torch.utils.data.Subset(full_train, range(cut, len(full_train)))
        self._test_raw  = torchvision.datasets.CIFAR10(
            root=self.datasets_path, train=False, download=download, transform=None,
        )

    def get_prediction_mode(self):
        return ClassificationMode(10)

    def raw_datasets(self) -> dict:
        return {"train": self._train_raw, "val": self._val_raw, "test": self._test_raw}

    def torch_transforms(self) -> dict:
        return {
            "train": [
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
            ],
            "val":  [transforms.ToTensor()],
            "test": [transforms.ToTensor()],
        }

    def ffcv_transforms(self) -> dict:
        # Augmentation only — spec_builder synthesizes the NormalizeImage +
        # ToTensor/ToDevice/ToTorchImage tail from _preprocessing_spec.
        # Strong CIFAR augmentation as a best-effort AutoAugment substitute:
        # HFlip + ±4-px translate ≡ RandomCrop(32, padding=4), 16×16 cutout,
        # brightness / contrast / saturation jitter.
        return {
            "train": [
                ("RandomHorizontalFlip", {}),
                ("RandomTranslate", {"padding": 4}),
                ("Cutout", {"crop_size": 16}),
                ("RandomBrightness", {"magnitude": 0.3}),
                ("RandomContrast", {"magnitude": 0.3}),
                ("RandomSaturation", {"magnitude": 0.3}),
            ],
            "val":  [],
            "test": [],
        }
