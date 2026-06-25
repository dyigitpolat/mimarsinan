"""ImageNet (ILSVRC2012) classification via `torchvision.datasets.ImageNet`."""

import os
from pathlib import Path

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


def _seeded_val_test_indices(n_official_val: int, val_fraction: float, seed: int | None):
    """Deterministic seeded partition of the OFFICIAL val into (gate-val, test).

    Both slices come from ``split="val"`` (genuinely held out for BOTH the
    from_scratch and pretrained regimes — neither slice is ever a train image).
    Official val is class-sorted, so a contiguous slice is class-starved; a
    seed-derived permutation makes the gate slice a representative sample over
    all 1000 classes, disjoint from the test slice, reproducible per seed.
    """
    g = torch.Generator()
    g.manual_seed(0 if seed is None else int(seed))
    perm = torch.randperm(n_official_val, generator=g).tolist()
    val_length = int(n_official_val * val_fraction)
    return perm[:val_length], perm[val_length:]


@BasicDataProviderFactory.register("ImageNet_DataProvider")
class ImageNet_DataProvider(DataProvider):
    """ILSVRC 2012 classification: 1000 classes.

    With ``IMAGENET_ROOT`` in ``.env``, creates ``<datasets_path>/imagenet`` as a symlink
    to that directory and uses it as the torchvision ``ImageNet`` root.

    **Train** is the FULL official ``split="train"`` (all 1000 classes) — used for any
    from_scratch training / fine-tune.

    **Validation** (the deployment TUNER GATE) is a seeded slice of the OFFICIAL
    ``split="val"`` (``official_val_fraction``, default 0.2). A train holdout would
    LEAK: a model trained on the full train (from_scratch) or any pretrained
    checkpoint has seen every train image, so a train-sourced val scores memorized
    accuracy. Sourcing the gate from the official val keeps it genuinely held out
    for BOTH regimes.

    **Test** (the final reported number) is the DISJOINT remaining slice of the
    OFFICIAL ``split="val"``. val and test are seeded, disjoint partitions of
    ``split="val"`` — never from train.
    """

    DISPLAY_LABEL = "ImageNet (224×224×3, 1000 classes)"
    SUPPORTS_PREPROCESSING = False

    # Fraction of the 50k official val reserved for the tuner GATE; the rest is the
    # reported-test slice. 0.2 -> ~10k gate (10/class, enough signal to gate
    # adaptation) + 40k test (robust reported number). Both stay disjoint held-out.
    official_val_fraction = 0.2

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        # ImageNet's crop policy is split-asymmetric — train uses
        # RandomResizedCrop(224), val uses Resize(256)+CenterCrop(224) —
        # so ``_preprocessing_spec.resize_to`` (single value, shared
        # across splits) can't express it. The provider declares the
        # crops itself in ``torch_transforms`` / ``ffcv_image_decoder``.
        # Normalize is symmetric across splits, so it goes via
        # ``_preprocessing_spec`` and the base class appends it uniformly.
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

        # Raw datasets (transforms applied via torch_transforms()).
        # Train = the FULL official train. The tuner GATE (val) and the reported
        # test are seeded, DISJOINT slices of the OFFICIAL val — never from train,
        # so the gate is genuinely held out for both from_scratch and pretrained.
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
        """The FULL official train (all 1000 classes), unwrapping any Subset.

        SSOT for from_scratch training: train is currently the raw train, but
        callers must not depend on whether it is wrapped.
        """
        train = self._train_raw
        return train.dataset if isinstance(train, torch.utils.data.Subset) else train

    def full_official_val_dataset(self):
        """The FULL 50k official val (all 1000 classes), unwrapping the Subset.

        ``val`` and ``test`` are disjoint slices of this; the from_scratch run
        reports the headline number on the full official val.
        """
        val = self._val_raw
        return val.dataset if isinstance(val, torch.utils.data.Subset) else val

    def torch_transforms(self) -> dict:
        # Normalize is appended uniformly by the base class from
        # ``_preprocessing_spec``; only the crop/resize policy lives here.
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
        # Per-split decoder carries the split-asymmetric crop policy:
        # RandomResizedCrop on train (matches torch path's training
        # recipe), CenterCrop on val/test (matches the deterministic
        # eval recipe). The decoder is the first FFCV op in each split.
        # beton_image_size=256 gives the train decoder room to sample
        # 224-px windows with scale/ratio variety.
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
        """Optimized torchvision fallback loaders for the fast-recipe trainer.

        The non-FFCV path: builds plain ``torch.utils.data.DataLoader`` over the
        provider's already-composed (transform-wrapped) splits with the
        throughput knobs FFCV would otherwise provide — many ``num_workers``,
        ``pin_memory``, ``persistent_workers``, and ``prefetch_factor``. Train is
        shuffled; val/test are sequential. This only assembles dataloaders; it is
        NOT a pipeline step and does not change the torch-DataLoader path used by
        the rest of the framework.
        """
        from torch.utils.data import DataLoader

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
