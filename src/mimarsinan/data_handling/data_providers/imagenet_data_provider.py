"""ImageNet (ILSVRC2012) classification via `torchvision.datasets.ImageNet`."""

import os
from pathlib import Path

import torchvision
import torchvision.transforms as transforms

from mimarsinan.data_handling.data_provider import ClassificationMode, DataProvider
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

_IMAGENET_LINK_NAME = "imagenet"
# .../src/mimarsinan/data_handling/data_providers/imagenet_data_provider.py -> repo root
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


@BasicDataProviderFactory.register("ImageNet_DataProvider")
class ImageNet_DataProvider(DataProvider):
    """ILSVRC 2012 classification: train split, val split, 1000 classes.

    With ``IMAGENET_ROOT`` in ``.env``, creates ``<datasets_path>/imagenet`` as a symlink
    to that directory and uses it as the torchvision ``ImageNet`` root. Otherwise
    ``datasets_path`` is the root. See `torchvision.datasets.ImageNet`.
    """

    DISPLAY_LABEL = "ImageNet (224×224×3, 1000 classes)"

    def __init__(self, datasets_path, *, seed: int | None = 0):
        super().__init__(datasets_path, seed=seed)

        root = _ensure_imagenet_symlink(str(self.datasets_path))
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"ImageNet root directory not found: {root!r}. "
                "Set IMAGENET_ROOT in .env (symlink under datasets_path) or use a valid datasets_path."
            )
        devkit = os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")
        meta = os.path.join(root, "meta.bin")
        if not os.path.isfile(meta) and not os.path.isfile(devkit):
            raise FileNotFoundError(
                f"ImageNet metadata missing: expected {meta!r} or {devkit!r} under {root!r}."
            )

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

        self.training_dataset = torchvision.datasets.ImageNet(
            root=root, split="train", transform=train_transform
        )
        self.validation_dataset = torchvision.datasets.ImageNet(
            root=root, split="val", transform=eval_transform
        )
        # No public test labels for ILSVRC2012; reuse validation for test metrics.
        self.test_dataset = self.validation_dataset

    def _get_training_dataset(self):
        return self.training_dataset

    def _get_validation_dataset(self):
        return self.validation_dataset

    def _get_test_dataset(self):
        return self.test_dataset

    def get_training_batch_size(self):
        return 256

    def get_validation_batch_size(self):
        return 256

    def get_test_batch_size(self):
        return 256

    def get_prediction_mode(self):
        return ClassificationMode(1000)
