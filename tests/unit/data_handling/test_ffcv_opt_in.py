"""Per-provider FFCV opt-in contract via the ``ffcv_transforms()`` override.

DataProvider's data surface is four method overrides:

* ``raw_datasets()``           — per-split raw datasets shared by both paths
* ``torch_transforms()``       — per-split raw torchvision transform lists
                                 for the torch DataLoader path
* ``ffcv_transforms()``        — per-split FFCV CPU op chains
                                 ``[(op_name, kwargs), ...]``. Non-empty opts
                                 in; the FFCV path is then a hard requirement.
* ``ffcv_image_field_kwargs()``— ``RGBImageField`` write kwargs
                                 (``max_resolution`` etc.).

``enable_ffcv()`` is derived: ``bool(self.ffcv_transforms())``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestEnableFFCVDerivedFromFFCVTransforms:
    def test_default_means_opt_out(self):
        from mimarsinan.data_handling.data_provider import DataProvider

        class _Bare(DataProvider):
            def __init__(self): pass

        p = _Bare()
        assert p.ffcv_transforms() == {}
        assert p.enable_ffcv() is False

    def test_override_with_populated_dict_opts_in(self):
        from mimarsinan.data_handling.data_provider import DataProvider

        class _Opted(DataProvider):
            def __init__(self): pass
            def ffcv_transforms(self):
                return {"splits": {"train": [("SimpleRGBImageDecoder", {})],
                                   "val":   [("SimpleRGBImageDecoder", {})],
                                   "test":  [("SimpleRGBImageDecoder", {})]}}

        assert _Opted().enable_ffcv() is True


class TestProvidersOptInExplicitly:
    """Providers that ship FFCV-ready: their ``ffcv_transforms()`` returns
    a structured ``{'beton_image_size'?: int, 'splits': {...}}`` dict and
    ``enable_ffcv()`` is True."""

    def test_cifar10_declares_strong_augmentation_chain(self):
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
            cfg = p.ffcv_transforms()
            assert "splits" in cfg
            assert set(cfg["splits"].keys()) == {"train", "val", "test"}
            train_names = [op[0] for op in cfg["splits"]["train"]]
            # Decoder is the first op.
            assert train_names[0] == "SimpleRGBImageDecoder"
            # Augments follow.
            assert "RandomHorizontalFlip" in train_names
            assert "RandomTranslate" in train_names
            assert "Cutout" in train_names
            # NormalizeImage is synthesized; not in the provider's chain.
            assert "NormalizeImage" not in train_names
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
            train_ops = p.ffcv_transforms()["splits"]["train"]
            train_names = [op[0] for op in train_ops]
            assert train_names[0] == "SimpleRGBImageDecoder"
            assert "RandomHorizontalFlip" not in train_names
            assert "RandomTranslate" in train_names
            assert "Cutout" in train_names
            assert "NormalizeImage" not in train_names

    def test_imagenet_declares_per_split_decoder_and_beton_size(self):
        """ImageNet's split-asymmetric crop policy lives in per-split
        decoders inside ``ffcv_transforms()``. ``beton_image_size=256`` so
        the train decoder has source pixels to crop from."""
        from mimarsinan.data_handling.data_provider import DataProvider
        from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
            ImageNet_DataProvider,
        )
        # Class-level inspection (avoids needing the ImageNet metadata files).
        assert ImageNet_DataProvider.ffcv_transforms is not DataProvider.ffcv_transforms
        cfg = ImageNet_DataProvider.ffcv_transforms(SimpleNamespace())
        assert cfg["beton_image_size"] == 256
        train_first = cfg["splits"]["train"][0]
        val_first   = cfg["splits"]["val"][0]
        test_first  = cfg["splits"]["test"][0]
        assert train_first[0] == "RandomResizedCropRGBImageDecoder"
        assert val_first[0]   == "CenterCropRGBImageDecoder"
        assert test_first[0]  == "CenterCropRGBImageDecoder"
        assert train_first[1]["output_size"] == (224, 224)
        assert val_first[1]["output_size"]   == (224, 224)


class TestProvidersOptOut:
    """Providers that don't ship FFCV-ready inherit the empty default."""

    def test_mnist_stays_opt_out(self):
        """FFCV's RGBImageField requires 3 channels and no stock op
        collapses back to 1, so MNIST stays on the torch path."""
        from mimarsinan.data_handling.data_provider import DataProvider
        from mimarsinan.data_handling.data_providers.mnist_data_provider import (
            MNIST_DataProvider,
        )
        assert MNIST_DataProvider.ffcv_transforms is DataProvider.ffcv_transforms

    def test_mnist32_stays_opt_out(self):
        from mimarsinan.data_handling.data_provider import DataProvider
        from mimarsinan.data_handling.data_providers.mnist32_data_provider import (
            MNIST32_DataProvider,
        )
        assert MNIST32_DataProvider.ffcv_transforms is DataProvider.ffcv_transforms

    def test_ecg_stays_opt_out(self):
        """ECG is 1D signal data; FFCV's RGBImageField doesn't apply."""
        from mimarsinan.data_handling.data_provider import DataProvider
        from mimarsinan.data_handling.data_providers.ecg_data_provider import (
            ECG_DataProvider,
        )
        assert ECG_DataProvider.ffcv_transforms is DataProvider.ffcv_transforms


class TestLoaderFactoryHonorsOptIn:
    """``_ffcv_loader`` returns None when the provider opts out, and on
    opt-in any FFCV-side error propagates (no silent fallback)."""

    def _make_factory(self):
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
        return DataLoaderFactory(
            data_provider_factory=MagicMock(),
            num_workers=0,
        )

    def test_opt_out_provider_skips_ffcv(self):
        factory = self._make_factory()
        provider = SimpleNamespace(enable_ffcv=lambda: False)
        assert factory._ffcv_loader("train", batch_size=4, data_provider=provider) is None

    def test_opt_in_provider_dispatches_to_ffcv_factory(self):
        factory = self._make_factory()
        # Stub out the FFCV factory creation so we don't need ffcv installed
        # to exercise the dispatch path.
        fake_ffcv = MagicMock()
        fake_ffcv.create_training_loader.return_value = "FFCV_TRAIN"
        fake_ffcv.create_validation_loader.return_value = "FFCV_VAL"
        fake_ffcv.create_test_loader.return_value = "FFCV_TEST"
        factory._ffcv_factory = fake_ffcv

        provider = SimpleNamespace(enable_ffcv=lambda: True)
        assert factory._ffcv_loader("train", batch_size=4, data_provider=provider) == "FFCV_TRAIN"
        assert factory._ffcv_loader("val",   batch_size=4, data_provider=provider) == "FFCV_VAL"
        assert factory._ffcv_loader("test",  batch_size=4, data_provider=provider) == "FFCV_TEST"

    def test_unknown_split_kind_raises(self):
        factory = self._make_factory()
        factory._ffcv_factory = MagicMock()
        provider = SimpleNamespace(enable_ffcv=lambda: True)
        with pytest.raises(ValueError, match="unknown split"):
            factory._ffcv_loader("foo", batch_size=4, data_provider=provider)
