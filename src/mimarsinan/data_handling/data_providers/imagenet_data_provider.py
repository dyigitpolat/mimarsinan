"""ImageNet (ILSVRC2012) classification via `torchvision.datasets.ImageNet`."""

import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mimarsinan.common.env import imagenet_root
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
    target = imagenet_root()
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


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _seeded_val_test_indices(n_official_val: int, val_fraction: float, seed: int | None):
    """Deterministic seeded partition of the official val into disjoint (gate-val, test) slices.

    A seed-derived permutation makes the class-sorted official val
    representative across all classes; both slices are reproducible per seed.
    """
    g = torch.Generator()
    g.manual_seed(0 if seed is None else int(seed))
    perm = torch.randperm(n_official_val, generator=g).tolist()
    val_length = int(n_official_val * val_fraction)
    return perm[:val_length], perm[val_length:]


@BasicDataProviderFactory.register("ImageNet_DataProvider")
class ImageNet_DataProvider(DataProvider):
    """ILSVRC 2012 classification (1000 classes).

    Train is the full official train; val (the tuner gate) and test are
    seeded, disjoint slices of the official val — never sourced from train.
    """

    DISPLAY_LABEL = "ImageNet (224×224×3, 1000 classes)"
    SUPPORTS_PREPROCESSING = False

    official_val_fraction = 0.2

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        super().__init__(
            datasets_path, seed=seed,
            preprocessing={"normalize": {"mean": _IMAGENET_MEAN, "std": _IMAGENET_STD}},
            batch_size=batch_size,
        )

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

        self._train_raw = torchvision.datasets.ImageNet(root=root, split="train", transform=None)

        official_val = torchvision.datasets.ImageNet(root=root, split="val", transform=None)
        val_idx, test_idx = _seeded_val_test_indices(
            len(official_val), self.official_val_fraction, self.seed
        )
        self._val_raw  = torch.utils.data.Subset(official_val, val_idx)
        self._test_raw = torch.utils.data.Subset(official_val, test_idx)

    def raw_datasets(self) -> dict:
        return {"train": self._train_raw, "val": self._val_raw, "test": self._test_raw}

    def full_train_dataset(self):
        """The full official train (all 1000 classes), unwrapping any Subset."""
        train = self._train_raw
        return train.dataset if isinstance(train, torch.utils.data.Subset) else train

    def full_official_val_dataset(self):
        """The full 50k official val (all 1000 classes), unwrapping the Subset."""
        val = self._val_raw
        return val.dataset if isinstance(val, torch.utils.data.Subset) else val

    def torch_transforms(self) -> dict:
        return {
            "train": [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ],
            "val": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ],
            "test": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ],
        }

    def ffcv_transforms(self) -> dict:
        return {
            "beton_image_size": 256,
            "splits": {
                "train": [
                    ("RandomResizedCropRGBImageDecoder", {
                        "output_size": (224, 224),
                        "scale": (0.08, 1.0),
                        "ratio": (3.0 / 4.0, 4.0 / 3.0),
                    }),
                    ("RandomHorizontalFlip", {}),
                ],
                "val": [
                    ("CenterCropRGBImageDecoder", {
                        "output_size": (224, 224),
                        "ratio": 224.0 / 256.0,
                    }),
                ],
                "test": [
                    ("CenterCropRGBImageDecoder", {
                        "output_size": (224, 224),
                        "ratio": 224.0 / 256.0,
                    }),
                ],
            },
        }

    def get_training_batch_size(self):
        return self._batch_size_override or 16

    def get_validation_batch_size(self):
        return self._batch_size_override or 16

    def get_test_batch_size(self):
        return self._batch_size_override or 16

    def get_prediction_mode(self):
        return ClassificationMode(1000)

    def fast_fallback_dataloaders(self, *, batch_size: int, num_workers: int = 12):
        """Plain torchvision DataLoaders (shuffled train, sequential val/test) for the fast-recipe trainer."""
        datasets = {
            "train": (self._get_training_dataset(), True),
            "val": (self._get_validation_dataset(), False),
            "test": (self._get_test_dataset(), False),
        }
        workers = max(0, int(num_workers))
        loader_kwargs = dict(
            batch_size=int(batch_size),
            num_workers=workers,
            pin_memory=True,
        )
        if workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        loaders = {}
        for split, (dataset, shuffle) in datasets.items():
            loaders[split] = DataLoader(
                dataset, shuffle=shuffle, drop_last=shuffle, **loader_kwargs
            )
        return loaders
