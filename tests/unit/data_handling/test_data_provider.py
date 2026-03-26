"""Tests for DataProvider, DataProviderFactory, and DataLoaderFactory."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode, RegressionMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from conftest import TinyDataProvider, MockDataProviderFactory


class TestClassificationMode:
    def test_mode(self):
        cm = ClassificationMode(10)
        assert cm.mode() == "classification"
        assert cm.num_classes == 10

    def test_create_loss(self):
        cm = ClassificationMode(10)
        loss = cm.create_loss()
        assert callable(loss)


class TestRegressionMode:
    def test_mode(self):
        rm = RegressionMode()
        assert rm.mode() == "regression"

    def test_create_loss(self):
        rm = RegressionMode()
        loss = rm.create_loss()
        assert callable(loss)


class TestTinyDataProvider:
    def test_get_datasets(self):
        dp = TinyDataProvider()
        train = dp._get_training_dataset()
        val = dp._get_validation_dataset()
        test = dp._get_test_dataset()
        assert len(train) == 10
        assert len(val) == 10
        assert len(test) == 10

    def test_get_input_shape(self):
        dp = TinyDataProvider(input_shape=(3, 4, 4))
        assert dp.get_input_shape() == (3, 4, 4)

    def test_get_output_shape(self):
        dp = TinyDataProvider(num_classes=5)
        assert dp.get_output_shape() == 5

    def test_prediction_mode(self):
        dp = TinyDataProvider(num_classes=4)
        pm = dp.get_prediction_mode()
        assert isinstance(pm, ClassificationMode)
        assert pm.num_classes == 4

    def test_batch_sizes(self):
        dp = TinyDataProvider(size=20)
        assert dp.get_training_batch_size() <= 20
        assert dp.get_validation_batch_size() == 20
        assert dp.get_test_batch_size() == 20

    def test_set_sizes(self):
        dp = TinyDataProvider(size=15)
        assert dp.get_training_set_size() == 15

    def test_custom_shape(self):
        dp = TinyDataProvider(input_shape=(1, 16, 16), num_classes=10)
        x, y = dp._get_training_dataset()[0]
        assert x.shape == (1, 16, 16)
        assert 0 <= y.item() < 10


class TestBasicDataProviderFactory:
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            BasicDataProviderFactory("nonexistent_provider_xyz", "/tmp")

    def test_registered_provider_creates(self):
        @BasicDataProviderFactory.register("_test_provider_unique_42")
        class _TestProvider(DataProvider):
            def _get_training_dataset(self):
                return []
            def _get_validation_dataset(self):
                return []
            def _get_test_dataset(self):
                return []
            def get_prediction_mode(self):
                return ClassificationMode(2)

        factory = BasicDataProviderFactory("_test_provider_unique_42", "/tmp")
        provider = factory.create()
        assert isinstance(provider, _TestProvider)

        del BasicDataProviderFactory._provider_registry["_test_provider_unique_42"]

    def test_list_registered_returns_id_and_label(self):
        import mimarsinan.data_handling.data_providers  # noqa: F401 — populate registry
        result = BasicDataProviderFactory.list_registered()
        assert isinstance(result, list)
        assert len(result) >= 1
        ids = [e["id"] for e in result]
        assert "MNIST_DataProvider" in ids
        for entry in result:
            assert "id" in entry
            assert "label" in entry
        assert "ImageNet_DataProvider" in ids

    def test_creates_loaders(self):
        dp_factory = MockDataProviderFactory()
        dlf = DataLoaderFactory(dp_factory, num_workers=0)
        dp = dlf.create_data_provider()

        train_loader = dlf.create_training_loader(4, dp)
        val_loader = dlf.create_validation_loader(10, dp)
        test_loader = dlf.create_test_loader(10, dp)

        batch = next(iter(train_loader))
        assert len(batch) == 2
        x, y = batch
        assert x.shape[0] <= 4

    def test_data_provider_caching(self):
        dp_factory = MockDataProviderFactory()
        dlf = DataLoaderFactory(dp_factory, num_workers=0)
        dp1 = dlf.create_data_provider()
        dp2 = dlf.create_data_provider()
        assert dp1 is dp2


class TestImageNetDataProvider:
    @staticmethod
    def _skip_dotenv(monkeypatch):
        """Isolate from repo ``.env``; tests set ``IMAGENET_ROOT`` / ``datasets_path`` like callers."""
        monkeypatch.setattr(
            "mimarsinan.data_handling.data_providers.imagenet_data_provider._load_project_dotenv",
            lambda: None,
        )
        monkeypatch.delenv("IMAGENET_ROOT", raising=False)

    def test_missing_root_raises(self, tmp_path, monkeypatch):
        """No ``datasets`` dir yet and no ``IMAGENET_ROOT`` → same resolution as production."""
        from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
            ImageNet_DataProvider,
        )

        self._skip_dotenv(monkeypatch)
        datasets_dir = tmp_path / "datasets"
        assert not datasets_dir.exists()
        with pytest.raises(FileNotFoundError, match="ImageNet root"):
            ImageNet_DataProvider(str(datasets_dir))

    def test_missing_metadata_raises(self, tmp_path, monkeypatch):
        """Empty directory without devkit / ``meta.bin`` (no symlink)."""
        from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
            ImageNet_DataProvider,
        )

        self._skip_dotenv(monkeypatch)
        data_root = tmp_path / "data_root"
        data_root.mkdir()
        with pytest.raises(FileNotFoundError, match="metadata"):
            ImageNet_DataProvider(str(data_root))

    def test_instantiation_with_mocked_torchvision(self, tmp_path, monkeypatch):
        """``datasets_path`` + ``IMAGENET_ROOT`` → symlink under ``datasets``; same as ``.env`` flow."""
        from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
            ImageNet_DataProvider,
        )

        self._skip_dotenv(monkeypatch)
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        target = tmp_path / "imagenet_target"
        target.mkdir()
        (target / "meta.bin").write_bytes(b"x")
        monkeypatch.setenv("IMAGENET_ROOT", str(target))

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 10

        def _getitem(_self, _idx):
            return torch.zeros(3, 224, 224), 0

        mock_ds.__getitem__ = _getitem

        with patch(
            "mimarsinan.data_handling.data_providers.imagenet_data_provider.torchvision.datasets.ImageNet",
            return_value=mock_ds,
        ):
            dp = ImageNet_DataProvider(str(datasets_dir))

        link = datasets_dir / "imagenet"
        assert link.is_symlink()
        assert link.resolve() == target.resolve()

        assert dp.get_prediction_mode().num_classes == 1000
        assert dp.get_training_batch_size() == 16
        assert dp.get_validation_batch_size() == 16
        assert dp.get_test_batch_size() == 16
        assert dp.get_input_shape() == (3, 224, 224)
