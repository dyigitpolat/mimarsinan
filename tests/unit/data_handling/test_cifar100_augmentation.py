"""CIFAR training augmentation: flip on both loaders, pad-4 crop on the torch path.

Both CIFAR-100 paths used to train without horizontal flip, and both CIFAR torch
paths without the pad-4 random crop (audit_v42 mixer_and_priors.md, loader
defects). The declarations are read without constructing the provider, so no
dataset is touched.
"""

from types import SimpleNamespace

import torch
import torchvision.transforms as T

from mimarsinan.data_handling.data_providers.cifar10_data_provider import CIFAR10_DataProvider
from mimarsinan.data_handling.data_providers.cifar100_data_provider import CIFAR100_DataProvider


def _torch_split(provider_cls, split):
    return provider_cls.torch_transforms(SimpleNamespace())[split]


def _ffcv_names(provider_cls, split):
    return [op[0] for op in provider_cls.ffcv_transforms(SimpleNamespace())["splits"][split]]


class TestCifar100TorchPath:
    def test_train_has_pad4_crop_then_flip_before_autoaugment(self):
        ops = _torch_split(CIFAR100_DataProvider, "train")
        assert [type(op) for op in ops] == [
            T.RandomCrop, T.RandomHorizontalFlip, T.AutoAugment, T.ToTensor,
        ]
        crop = ops[0]
        assert tuple(crop.size) == (32, 32)
        assert crop.padding == 4
        assert ops[1].p == 0.5

    def test_eval_splits_stay_deterministic(self):
        for split in ("val", "test"):
            ops = _torch_split(CIFAR100_DataProvider, split)
            assert [type(op) for op in ops] == [T.ToTensor]

    def test_train_pipeline_keeps_shape_and_range(self):
        from PIL import Image
        import numpy as np

        torch.manual_seed(0)
        img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
        out = T.Compose(_torch_split(CIFAR100_DataProvider, "train"))(img)
        assert out.shape == (3, 32, 32)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_torch_train_chain_equals_cifar10(self):
        c10 = [type(op) for op in _torch_split(CIFAR10_DataProvider, "train")]
        c100 = [type(op) for op in _torch_split(CIFAR100_DataProvider, "train")]
        assert c10 == c100
        assert _torch_split(CIFAR10_DataProvider, "train")[0].padding == 4

    def test_flip_matches_cifar10_flip(self):
        c10 = [op for op in _torch_split(CIFAR10_DataProvider, "train") if isinstance(op, T.RandomHorizontalFlip)]
        c100 = [op for op in _torch_split(CIFAR100_DataProvider, "train") if isinstance(op, T.RandomHorizontalFlip)]
        assert len(c10) == len(c100) == 1
        assert c10[0].p == c100[0].p


class TestCifar100FfcvPath:
    def test_train_flips_like_cifar10(self):
        c100 = _ffcv_names(CIFAR100_DataProvider, "train")
        c10 = _ffcv_names(CIFAR10_DataProvider, "train")
        assert c100[:3] == c10[:3] == ["SimpleRGBImageDecoder", "RandomHorizontalFlip", "RandomTranslate"]

    def test_train_chain_equals_cifar10(self):
        assert _ffcv_names(CIFAR100_DataProvider, "train") == _ffcv_names(CIFAR10_DataProvider, "train")

    def test_eval_splits_decode_only(self):
        for split in ("val", "test"):
            assert _ffcv_names(CIFAR100_DataProvider, split) == ["SimpleRGBImageDecoder"]
