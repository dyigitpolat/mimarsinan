""

from __future__ import annotations

import json
from pathlib import Path

from mimarsinan.data_handling.data_provider import ClassificationMode, DataProvider
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

import w38common as c

PROVIDER_NAME = "W38_ImageNetSubset_DataProvider"
VIEW = "view.json"


class _ListedImages:
    ""

    def __init__(self, entries: list[dict]):
        self.entries = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        return c.load_image(e), int(e["class_index"])


def read_view(datasets_path) -> dict:
    path = Path(datasets_path) / VIEW
    if not path.is_file():
        raise SystemExit(f"REFUSED: {path} is missing: a W38 session reads its view file")
    return json.loads(path.read_text())


@BasicDataProviderFactory.register(PROVIDER_NAME)
class W38ImageNetSubsetProvider(DataProvider):
    DISPLAY_LABEL = "ImageNet W38 subset (manifest prefix; calibration from the training split)"
    RAW_INPUT_VALUE_RANGE = (0.0, 1.0)

    def __init__(self, datasets_path, *, seed=0, preprocessing=None, batch_size=None):
        super().__init__(datasets_path, seed=seed, preprocessing=preprocessing, batch_size=batch_size)
        c.install_data_guard()
        self.view = read_view(datasets_path)
        self.member = c.members()[self.view["member"]]
        spec = self._preprocessing_spec
        want = self.member["transforms"]
        if spec is None or list(spec.mean or ()) != want["mean"] or list(spec.std or ()) != want["std"] \
                or spec.resize_to is not None:
            raise SystemExit(f"REFUSED: preprocessing {spec} is not the normalization of {self.view['member']}'s "
                             "transforms() (resize and crop are the provider's)")
        test = c.split_entries(self.view["split"], int(self.view["test_images"]))
        cal = c.calibration_entries(int(self.view["calibration_images"]))
        self.entries = {"train": cal, "val": cal, "test": test}
        self._raw = {k: _ListedImages(v) for k, v in self.entries.items()}
        self.transform_check = {k: c.check_transform_equality(self.member, v[:4],
                                                              self._wrap_with_preprocessing(self._base()))
                                for k, v in (("test", test), ("val", cal))}
        if not all(v["passed"] for v in self.transform_check.values()):
            raise SystemExit(f"REFUSED: the composed preprocessing differs from transforms(): {self.transform_check}")

    def _base(self) -> list:
        return c.base_transforms(self.member)

    def get_prediction_mode(self):
        return ClassificationMode(1000)

    def raw_datasets(self) -> dict:
        return dict(self._raw)

    def torch_transforms(self) -> dict:
        return {k: self._base() for k in ("train", "val", "test")}

    def get_training_batch_size(self):
        return int(self.view["calibration_batch"])

    def get_validation_batch_size(self):
        return int(self.view["calibration_batch"])

    def get_test_batch_size(self):
        return int(self.view["test_batch"])

    def digests(self) -> dict:
        def listing(entries):
            return c.sha256_bytes("\n".join(f"{e['file']} {e['sha256']}" for e in entries).encode())

        return {"view": self.view, "manifest_sha256": c.sha256_file(c.MANIFEST),
                "dev_sha256": c.sha256_file(c.DEV) if c.DEV.exists() else None,
                "calibration_sha256": c.sha256_file(c.CALIBRATION),
                **{f"{k}_listing_sha256": listing(v) for k, v in self.entries.items()},
                "transform_check": self.transform_check}
