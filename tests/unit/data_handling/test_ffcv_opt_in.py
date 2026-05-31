"""Per-provider FFCV opt-in contract via the ``ffcv_transforms()`` override.

DataProvider's data surface is three method overrides:

* ``raw_datasets() -> dict``     — per-split raw datasets, shared by both paths.
* ``torch_transforms() -> dict`` — per-split torchvision ``Compose`` for the torch DataLoader path.
* ``ffcv_transforms() -> dict``  — per-split FFCV CPU op chain ``[(op_name, kwargs), ...]``.

Defaults return ``{}``. A provider opts into FFCV by overriding
``ffcv_transforms()`` to return a non-empty dict (even with empty per-split
lists). Providers that don't override it get the torch DataLoader path
regardless of ``MIMARSINAN_PERF_FFCV``. ``enable_ffcv()`` is the derived
gate: ``bool(self.ffcv_transforms())``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestEnableFFCVDerivedFromFFCVTransforms:
    def test_default_means_opt_out(self):
        from mimarsinan.data_handling.data_provider import DataProvider

        class _Bare(DataProvider):
            def __init__(self): pass  # skip super().__init__ for unit test

        p = _Bare()
        assert p.ffcv_transforms() == {}
        assert p.enable_ffcv() is False

    def test_override_with_populated_dict_opts_in(self):
        from mimarsinan.data_handling.data_provider import DataProvider

        class _Opted(DataProvider):
            def __init__(self): pass
            def ffcv_transforms(self):
                return {"train": [], "val": [], "test": []}

        p = _Opted()
        assert p.enable_ffcv() is True


class TestProvidersOptInExplicitly:
    """Providers that ship FFCV-ready: their ``ffcv_transforms()`` returns a
    populated dict."""

    def test_cifar10_declares_strong_augmentation_for_train_only(self):
        from mimarsinan.data_handling.data_providers.cifar10_data_provider import (
            CIFAR10_DataProvider,
        )
        pytest.importorskip("torchvision")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = CIFAR10_DataProvider(datasets_path=tmp)
            except Exception as e:
                pytest.skip(f"CIFAR10 download/init not viable: {e!r}")
            ffcv_tf = p.ffcv_transforms()
            assert set(ffcv_tf.keys()) == {"train", "val", "test"}
            train_names = [op[0] for op in ffcv_tf["train"]]
            assert "RandomHorizontalFlip" in train_names
            assert "RandomTranslate" in train_names
            assert "Cutout" in train_names
            assert "RandomBrightness" in train_names
            assert ffcv_tf["val"] == []
            assert ffcv_tf["test"] == []
            assert p.enable_ffcv() is True

    def test_cifar100_declares_strong_augmentation_no_flip(self):
        from mimarsinan.data_handling.data_providers.cifar100_data_provider import (
            CIFAR100_DataProvider,
        )
        pytest.importorskip("torchvision")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = CIFAR100_DataProvider(datasets_path=tmp)
            except Exception as e:
                pytest.skip(f"CIFAR100 download/init not viable: {e!r}")
            ffcv_tf = p.ffcv_transforms()
            train_names = [op[0] for op in ffcv_tf["train"]]
            assert "RandomHorizontalFlip" not in train_names
            assert "RandomTranslate" in train_names
            assert "Cutout" in train_names
            assert ffcv_tf["val"] == []
            assert p.enable_ffcv() is True

    def test_mnist_declares_empty_chains(self):
        from mimarsinan.data_handling.data_providers.mnist_data_provider import (
            MNIST_DataProvider,
        )
        pytest.importorskip("torchvision")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = MNIST_DataProvider(datasets_path=tmp)
            except Exception as e:
                pytest.skip(f"MNIST download/init not viable: {e!r}")
            assert p.ffcv_transforms() == {"train": [], "val": [], "test": []}
            assert p.enable_ffcv() is True

    def test_mnist32_declares_empty_chains_and_resize_to_32(self):
        """MNIST32 opts into FFCV with no augments; the 28→32 upscale runs
        uniformly via the GPU postprocess driven by
        ``_preprocessing_spec.resize_to``."""
        from mimarsinan.data_handling.data_providers.mnist32_data_provider import (
            MNIST32_DataProvider,
        )
        pytest.importorskip("torchvision")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = MNIST32_DataProvider(datasets_path=tmp)
            except Exception as e:
                pytest.skip(f"MNIST32 download/init not viable: {e!r}")
            assert p.ffcv_transforms() == {"train": [], "val": [], "test": []}
            assert p.enable_ffcv() is True
            assert p._preprocessing_spec is not None
            assert p._preprocessing_spec.resize_to == 32


class TestProvidersOptOut:
    """Providers that haven't been wired for FFCV inherit the empty default."""

    def test_imagenet_stays_opt_out(self):
        from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
            ImageNet_DataProvider,
        )
        # Inspect the class-level default by binding to a minimal namespace.
        # ImageNet doesn't override ffcv_transforms (RandomResizedCrop +
        # variable-size source images need a beton-writer refactor first).
        from mimarsinan.data_handling.data_provider import DataProvider
        # Class MRO check: ImageNet doesn't override ffcv_transforms.
        assert ImageNet_DataProvider.ffcv_transforms is DataProvider.ffcv_transforms

    def test_ecg_stays_opt_out(self):
        from mimarsinan.data_handling.data_providers.ecg_data_provider import (
            ECG_DataProvider,
        )
        from mimarsinan.data_handling.data_provider import DataProvider
        # ECG is 1D signal data, not images. FFCV's RGBImageField doesn't
        # apply; provider doesn't override ffcv_transforms.
        assert ECG_DataProvider.ffcv_transforms is DataProvider.ffcv_transforms


class TestLoaderFactoryHonorsOptIn:
    """``_try_ffcv`` returns None when the provider opts out, even if FFCV
    itself is installed and globally enabled."""

    def _make_factory(self, ffcv_available):
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

        fake_data_provider_factory = MagicMock()
        f = DataLoaderFactory(
            data_provider_factory=fake_data_provider_factory,
            num_workers=0,
        )
        if ffcv_available:
            ffcv_factory = MagicMock()
            ffcv_factory.create_training_loader.return_value = "FFCV_TRAIN"
            ffcv_factory.create_validation_loader.return_value = "FFCV_VAL"
            ffcv_factory.create_test_loader.return_value = "FFCV_TEST"
            f._ffcv_factory = ffcv_factory
            f._ffcv = lambda: ffcv_factory
        else:
            f._ffcv = lambda: None
        return f

    def test_opt_out_provider_skips_ffcv_even_when_factory_available(self):
        factory = self._make_factory(ffcv_available=True)
        provider = SimpleNamespace(enable_ffcv=lambda: False)
        assert factory._try_ffcv("train", batch_size=4, data_provider=provider) is None

    def test_opt_in_provider_uses_ffcv_when_factory_available(self):
        factory = self._make_factory(ffcv_available=True)
        provider = SimpleNamespace(enable_ffcv=lambda: True)
        assert factory._try_ffcv("train", batch_size=4, data_provider=provider) == "FFCV_TRAIN"
        assert factory._try_ffcv("val", batch_size=4, data_provider=provider) == "FFCV_VAL"
        assert factory._try_ffcv("test", batch_size=4, data_provider=provider) == "FFCV_TEST"

    def test_opt_in_provider_with_no_factory_returns_none(self):
        factory = self._make_factory(ffcv_available=False)
        provider = SimpleNamespace(enable_ffcv=lambda: True)
        assert factory._try_ffcv("train", batch_size=4, data_provider=provider) is None
