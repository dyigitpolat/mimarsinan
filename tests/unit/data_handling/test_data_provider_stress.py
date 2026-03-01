"""
Stress tests for DataProvider and DataLoaderFactory.

Tests edge cases: empty datasets, single-sample datasets, mismatched shapes.
"""

import pytest
import torch

from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from conftest import TinyDataset, TinyDataProvider, MockDataProviderFactory


class TestTinyDatasetEdgeCases:
    def test_single_sample(self):
        ds = TinyDataset(size=1)
        x, y = ds[0]
        assert x.shape == (1, 8, 8)
        assert 0 <= y.item() < 4

    def test_single_class(self):
        """Only 1 class — all targets should be 0."""
        ds = TinyDataset(num_classes=1, size=10)
        for i in range(len(ds)):
            _, y = ds[i]
            assert y.item() == 0

    def test_high_dimensional_input(self):
        ds = TinyDataset(input_shape=(3, 32, 32), size=5)
        x, _ = ds[0]
        assert x.shape == (3, 32, 32)

    def test_flat_input(self):
        """1D input shape (no spatial dims)."""
        ds = TinyDataset(input_shape=(100,), size=5)
        x, _ = ds[0]
        assert x.shape == (100,)


class TestDataLoaderFactoryStress:
    def test_iterate_multiple_epochs(self):
        """Verify data loader can be iterated multiple times."""
        dp_factory = MockDataProviderFactory()
        dlf = DataLoaderFactory(dp_factory, num_workers=0)
        dp = dlf.create_data_provider()
        loader = dlf.create_training_loader(4, dp)

        for epoch in range(3):
            batch_count = 0
            for x, y in loader:
                batch_count += 1
                assert x.shape[0] <= 4
            assert batch_count > 0

    def test_create_loaders_with_batch_size_one(self):
        dp_factory = MockDataProviderFactory()
        dlf = DataLoaderFactory(dp_factory, num_workers=0)
        dp = dlf.create_data_provider()

        loader = dlf.create_training_loader(1, dp)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 1

    def test_data_provider_shapes_consistent(self):
        """Input shape from provider should match actual data shapes."""
        dp = TinyDataProvider(input_shape=(3, 16, 16), num_classes=10)
        reported_shape = dp.get_input_shape()
        x, _ = dp._get_training_dataset()[0]
        actual_shape = tuple(x.shape)
        assert reported_shape == actual_shape

    def test_prediction_mode_num_classes(self):
        """Prediction mode should report the correct number of classes."""
        for nc in [2, 5, 10, 100]:
            dp = TinyDataProvider(num_classes=nc)
            pm = dp.get_prediction_mode()
            assert pm.num_classes == nc
