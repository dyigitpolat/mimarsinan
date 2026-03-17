"""Tests for shutdown_data_loader helper."""

import pytest

from mimarsinan.data_handling.data_loader_factory import (
    DataLoaderFactory,
    shutdown_data_loader,
)

from conftest import TinyDataProvider, MockDataProviderFactory


class TestShutdownDataLoader:
    """Test shutdown_data_loader with multi-worker DataLoaders."""

    def test_shutdown_none_is_noop(self):
        shutdown_data_loader(None)

    def test_shutdown_zero_workers_loader_is_noop(self):
        dp_factory = MockDataProviderFactory()
        dlf = DataLoaderFactory(dp_factory, num_workers=0)
        dp = dlf.create_data_provider()
        loader = dlf.create_training_loader(4, dp)
        shutdown_data_loader(loader)
        # Should not raise; loader still usable for iteration if needed
        list(loader)

    def test_shutdown_after_partial_iteration_clears_iterator(self):
        dp_factory = MockDataProviderFactory()
        dlf = DataLoaderFactory(dp_factory, num_workers=1)
        dp = dlf.create_data_provider()
        loader = dlf.create_training_loader(2, dp)
        next(iter(loader))
        assert loader._iterator is not None
        shutdown_data_loader(loader)
        assert loader._iterator is None

    def test_shutdown_idempotent(self):
        dp_factory = MockDataProviderFactory()
        dlf = DataLoaderFactory(dp_factory, num_workers=1)
        dp = dlf.create_data_provider()
        loader = dlf.create_training_loader(2, dp)
        next(iter(loader))
        shutdown_data_loader(loader)
        shutdown_data_loader(loader)
        # Second call must not raise
