"""ImageNet (ILSVRC2012) classification via `torchvision.datasets.ImageNet`."""

import os
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from mimarsinan.data_handling.data_provider import ClassificationMode, DataProvider
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

_IMAGENET_LINK_NAME = "imagenet"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env")


def _ensure_imagenet_symlink(datasets_path: str) -> str:
    """Return torchvision ``root``: ``datasets_path/imagenet`` symlinked to ``IMAGENET_ROOT``."""
    _load_project_dotenv()
    target = os.environ.get("IMAGENET_ROOT", "").strip()
    if not target:
        return os.path.abspath(os.path.expanduser(str(datasets_path)))

    target = os.path.abspath(os.path.expanduser(target))
    if not os.path.isdir(target):
        raise FileNotFoundError(f"IMAGENET_ROOT is not a directory: {target!r}")

    base = os.path.abspath(os.path.expanduser(str(datasets_path)))
    if os.path.normpath(target) == os.path.normpath(base):
        return base

    os.makedirs(base, exist_ok=True)
    link = os.path.join(base, _IMAGENET_LINK_NAME)
    if os.path.islink(link):
        if os.path.realpath(link) == target:
            return link
        os.unlink(link)
        os.symlink(target, link)
    elif os.path.lexists(link):
        raise FileNotFoundError(
            f"Cannot create ImageNet symlink {link!r}: path exists and is not a symlink."
        )
    else:
        os.symlink(target, link)

    return link


# ImageNet normalization used with pretrained checkpoints.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_MEAN_255 = np.array(_IMAGENET_MEAN) * 255.0
_IMAGENET_STD_255  = np.array(_IMAGENET_STD)  * 255.0


@BasicDataProviderFactory.register("ImageNet_DataProvider")
class ImageNet_DataProvider(DataProvider):
    """ILSVRC 2012 classification: 1000 classes.

    With ``IMAGENET_ROOT`` in ``.env``, creates ``<datasets_path>/imagenet`` as a symlink
    to that directory and uses it as the torchvision ``ImageNet`` root.

    **Validation** is a tail of the **training** split (same convention as
    MNIST/CIFAR-10): ``training_validation_split`` (default 0.95) for training, remainder
    for validation.

    **Test** uses the official ``split="val"`` set (held-out; no public test labels in ILSVRC).
    """

    DISPLAY_LABEL = "ImageNet (224×224×3, 1000 classes)"
    SUPPORTS_PREPROCESSING = False

    training_validation_split = 0.95

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        # ImageNet supplies its own RandomResizedCrop / CenterCrop pipeline;
        # ignore the configured preprocessing so we don't double-resize.
        super().__init__(datasets_path, seed=seed, preprocessing=None, batch_size=batch_size)

        root = _ensure_imagenet_symlink(str(self.datasets_path))
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"ImageNet root directory not found: {root!r}. "
                "Set IMAGENET_ROOT in .env or use a valid datasets_path."
            )
        devkit = os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")
        meta = os.path.join(root, "meta.bin")
        if not os.path.isfile(meta) and not os.path.isfile(devkit):
            raise FileNotFoundError(
                f"ImageNet metadata missing: expected {meta!r} or {devkit!r} under {root!r}."
            )

        # Raw datasets (transforms applied via torch_transforms()).
        training_full = torchvision.datasets.ImageNet(root=root, split="train", transform=None)
        n_train = len(training_full)
        training_length = int(n_train * self.training_validation_split)

        self._train_raw = torch.utils.data.Subset(training_full, range(0, training_length))
        self._val_raw   = torch.utils.data.Subset(training_full, range(training_length, n_train))
        self._test_raw  = torchvision.datasets.ImageNet(root=root, split="val", transform=None)

    def raw_datasets(self) -> dict:
        return {"train": self._train_raw, "val": self._val_raw, "test": self._test_raw}

    def torch_transforms(self) -> dict:
        return {
            "train": [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ],
            "val": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ],
            "test": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ],
        }

    def ffcv_image_field_kwargs(self) -> dict:
        # Beton at 224 — matches the model input; train uses random crop +
        # flip at the FFCV decoder level (RandomResizedCropRGBImageDecoder).
        return {"max_resolution": 256}

    def ffcv_transforms(self) -> dict:
        normalize = ("NormalizeImage",
                     {"mean": _IMAGENET_MEAN_255, "std": _IMAGENET_STD_255, "type": np.float32})
        return {
            "train": [
                ("RandomHorizontalFlip", {}),
                ("ToTensor", {}),
                ("ToDevice", {"non_blocking": True}),
                ("ToTorchImage", {}),
                normalize,
            ],
            "val": [
                ("ToTensor", {}),
                ("ToDevice", {"non_blocking": True}),
                ("ToTorchImage", {}),
                normalize,
            ],
            "test": [
                ("ToTensor", {}),
                ("ToDevice", {"non_blocking": True}),
                ("ToTorchImage", {}),
                normalize,
            ],
        }

    def get_training_batch_size(self):
        return self._batch_size_override or 16

    def get_validation_batch_size(self):
        return self._batch_size_override or 16

    def get_test_batch_size(self):
        return self._batch_size_override or 16

    def get_prediction_mode(self):
        return ClassificationMode(1000)
