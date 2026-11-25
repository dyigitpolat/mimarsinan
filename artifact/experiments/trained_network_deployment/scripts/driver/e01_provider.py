""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from PIL import Image

from mimarsinan.data_handling.data_provider import ClassificationMode, DataProvider
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

import common

PROVIDER_NAME = "E01_CIFAR100_DataProvider"
TRAIN_MD5 = "16019d7e3df5f24257cddd939b257f8d"
SPLIT_FILE = common.B / "screen_v42" / "val_split_e01.json"
SPLIT_SHA256 = "f923919b8d1656f8f47ffce428b907c9366fab08316d71c4486a4f01f27809f2"
E01_INDEX_SHA256 = "bdc486b322b90bdb76e347544a09d3cc047cfbe75b8172466f02dd2244157337"
CALIBRATION_IMAGES = 2048
CALIBRATION_BATCH = 1024
CALIBRATION_SEED = 0
CALIBRATION_INDEX_SHA256 = "4e0e07fb97d5b2a3fa8f52b23da3e476ca54ccb6b5177a03508de94a3af48c09"
TRAINING_BATCH = 450
TEST_BATCH = 500


class HeldOutBatchRefusal(PermissionError):
    ""


def train_batch_path(datasets_path) -> Path:
    path = Path(datasets_path) / "cifar-100-python" / "train"
    for p in (path, path.resolve()):
        if p.name == "test":
            raise HeldOutBatchRefusal(f"REFUSED: {p} names the test batch")
    return path


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train(datasets_path) -> tuple[np.ndarray, np.ndarray]:
    ""
    path = train_batch_path(datasets_path)
    digest = _md5(path)
    if digest != TRAIN_MD5:
        raise SystemExit(f"REFUSED: {path} md5 {digest} != {TRAIN_MD5}")
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    images = np.asarray(entry["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    return images, np.asarray(entry["fine_labels"], dtype=np.int64)


def e01_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ""
    raw = SPLIT_FILE.read_bytes()
    if common.sha256_bytes(raw) != SPLIT_SHA256:
        raise SystemExit("REFUSED: split file sha256 differs")
    split = json.loads(raw)
    val = np.asarray(split["val_indices"], dtype=np.int64)
    if common.indices_sha256(val) != E01_INDEX_SHA256:
        raise SystemExit("REFUSED: split index digest differs")
    rng = np.random.default_rng(split["seed"])
    picks = [rng.permutation(np.flatnonzero(labels == c))[: split["per_class"]] for c in range(100)]
    if not np.array_equal(np.sort(np.concatenate(picks)), val):
        raise SystemExit("REFUSED: split differs from the declared stratified rule")
    mask = np.ones(len(labels), dtype=bool)
    mask[val] = False
    return np.flatnonzero(mask), val


def calibration_indices(train_idx: np.ndarray) -> np.ndarray:
    ""
    rng = np.random.default_rng(CALIBRATION_SEED + 1000)
    cal = np.asarray(train_idx)[rng.permutation(len(train_idx))[:CALIBRATION_IMAGES]]
    if common.indices_sha256(cal) != CALIBRATION_INDEX_SHA256:
        raise SystemExit("REFUSED: calibration draw differs from the gate's recorded draw")
    return cal


class ArraySplit:
    ""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.images, self.labels, self.indices = images, labels, np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        return Image.fromarray(self.images[j]), int(self.labels[j])


def train_transforms() -> list:
    ""
    return [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor()]


@BasicDataProviderFactory.register(PROVIDER_NAME)
class E01Cifar100Provider(DataProvider):
    DISPLAY_LABEL = "CIFAR-100 E01 (training batch only: 45,000 train, 2,048 calibration, 5,000 E01)"
    RAW_INPUT_VALUE_RANGE = (0.0, 1.0)

    def __init__(self, datasets_path, *, seed=0, preprocessing=None, batch_size=None):
        super().__init__(datasets_path, seed=seed, preprocessing=preprocessing, batch_size=batch_size)
        images, labels = load_train(datasets_path)
        train_idx, e01_idx = e01_split(labels)
        cal_idx = calibration_indices(train_idx)
        self.indices = {"train": train_idx, "val": cal_idx, "test": e01_idx}
        self._raw = {k: ArraySplit(images, labels, v) for k, v in self.indices.items()}

    def get_prediction_mode(self):
        return ClassificationMode(100)

    def raw_datasets(self) -> dict:
        return dict(self._raw)

    def torch_transforms(self) -> dict:
        return {"train": train_transforms(), "val": [T.ToTensor()], "test": [T.ToTensor()]}

    def get_training_batch_size(self):
        return self._batch_size_override or TRAINING_BATCH

    def get_validation_batch_size(self):
        return CALIBRATION_BATCH

    def get_test_batch_size(self):
        return TEST_BATCH

    def digests(self) -> dict:
        return {"train_md5": TRAIN_MD5, "split_sha256": SPLIT_SHA256,
                **{f"{k}_indices_sha256": common.indices_sha256(v) for k, v in self.indices.items()},
                "preprocessing": None if self._preprocessing_spec is None else
                {"mean": list(self._preprocessing_spec.mean or ()), "std": list(self._preprocessing_spec.std or ())}}


SMOKE_IMAGES = 64
SMOKE_PROVIDER_NAME = f"E01_CIFAR100_DataProvider_first{SMOKE_IMAGES}"


@BasicDataProviderFactory.register(SMOKE_PROVIDER_NAME)
class E01SmokeProvider(E01Cifar100Provider):
    DISPLAY_LABEL = f"CIFAR-100 E01 smoke (training batch only: the first {SMOKE_IMAGES} E01 images as test)"

    def __init__(self, datasets_path, *, seed=0, preprocessing=None, batch_size=None):
        super().__init__(datasets_path, seed=seed, preprocessing=preprocessing, batch_size=batch_size)
        base = self._raw["test"]
        self.indices["test"] = self.indices["test"][:SMOKE_IMAGES]
        self._raw["test"] = ArraySplit(base.images, base.labels, self.indices["test"])


PROVIDERS = {PROVIDER_NAME: E01Cifar100Provider, SMOKE_PROVIDER_NAME: E01SmokeProvider}
