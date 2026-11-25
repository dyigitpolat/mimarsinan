""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from screen_common import DATA_DIR, SPLIT_JSON, TRAIN_MD5, md5_file

NUM_CLASSES = 100
VAL_PER_CLASS = 50


def load_train_batch(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    ""
    path = Path(data_dir) / "train"
    digest = md5_file(path)
    if digest != TRAIN_MD5:
        raise SystemExit(f"REFUSED: {path} md5 {digest} != {TRAIN_MD5}")
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    images = np.asarray(entry["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    labels = np.asarray(entry["fine_labels"], dtype=np.int64)
    return images, labels


def stratified_split(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    ""
    rng = np.random.default_rng(seed)
    picks = [rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in range(NUM_CLASSES)]
    return np.sort(np.concatenate(picks))


def indices_sha256(indices) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in indices).encode()).hexdigest()


def load_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ""
    split = json.loads(Path(SPLIT_JSON).read_text())
    val = np.asarray(split["val_indices"], dtype=np.int64)
    if indices_sha256(val) != split["val_indices_sha256"]:
        raise SystemExit("REFUSED: split file indices do not match their recorded sha256")
    rederived = stratified_split(labels, split["per_class"], split["seed"])
    if not np.array_equal(rederived, val):
        raise SystemExit("REFUSED: split file differs from the declared stratified rule")
    mask = np.ones(len(labels), dtype=bool)
    mask[val] = False
    return np.flatnonzero(mask), val


class CifarArrayDataset(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray, transform):
        self.images, self.labels, self.indices, self.transform = images, labels, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        return self.transform(Image.fromarray(self.images[j])), int(self.labels[j])


def transforms_for(split: str):
    ""
    import torchvision.transforms as T
    from mimarsinan.data_handling.data_providers.cifar100_data_provider import (
        CIFAR100_MEAN,
        CIFAR100_STD,
        CIFAR100_DataProvider,
    )

    ops = CIFAR100_DataProvider.torch_transforms(object.__new__(CIFAR100_DataProvider))[split]
    return T.Compose(list(ops) + [T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])


def epoch_generator(seed: int, phase: str, epoch: int) -> torch.Generator:
    ""
    g = torch.Generator()
    g.manual_seed(int(hashlib.sha256(f"{seed}/{phase}/{epoch}".encode()).hexdigest()[:15], 16))
    return g


def make_loader(dataset, batch_size: int, *, shuffle: bool, generator, workers: int, drop_last: bool):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=workers,
        pin_memory=True, drop_last=drop_last, persistent_workers=False,
        prefetch_factor=4 if workers > 0 else None,
    )
