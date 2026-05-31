from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

import numpy as np
import torchvision.transforms as transforms
import torchvision

import torch
import os


_IMAGENET_MEAN_255 = np.array([0.485, 0.456, 0.406]) * 255.0
_IMAGENET_STD_255  = np.array([0.229, 0.224, 0.225]) * 255.0


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
        # Strong CIFAR augmentation — pre-AutoAugment baseline: HFlip + ±4-px
        # translate ≡ RandomCrop(32, padding=4) + 16×16 cutout + color jitter.
        # NormalizeImage runs CPU-side (before ToDevice) so we don't depend
        # on cupy for FFCV's GPU normalize path.
        normalize = self._normalize_op()
        return {
            "train": [
                ("RandomHorizontalFlip", {}),
                ("RandomTranslate", {"padding": 4}),
                ("Cutout", {"crop_size": 16}),
                ("RandomBrightness", {"magnitude": 0.3}),
                ("RandomContrast", {"magnitude": 0.3}),
                ("RandomSaturation", {"magnitude": 0.3}),
                normalize,
                ("ToTensor", {}),
                ("ToDevice", {"non_blocking": True}),
                ("ToTorchImage", {}),
            ],
            "val":  self._eval_chain(normalize),
            "test": self._eval_chain(normalize),
        }

    @staticmethod
    def _eval_chain(normalize):
        return [
            normalize,
            ("ToTensor", {}),
            ("ToDevice", {"non_blocking": True}),
            ("ToTorchImage", {}),
        ]

    def _normalize_op(self):
        spec = self._preprocessing_spec
        if spec is not None and spec.mean is not None and spec.std is not None:
            mean = np.array(spec.mean) * 255.0
            std  = np.array(spec.std)  * 255.0
        else:
            mean = _IMAGENET_MEAN_255
            std  = _IMAGENET_STD_255
        return ("NormalizeImage", {"mean": mean, "std": std, "type": np.float32})
