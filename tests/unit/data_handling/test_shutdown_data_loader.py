"""Tests for shutdown_data_loader helper."""

import logging

import pytest

from mimarsinan.data_handling.data_loader_factory import (
    DataLoaderFactory,
    _unregister_dataloader_atexit_handlers,
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


class TestShutdownDegradesViaBestEffort:
    """Shutdown-path failures degrade through the best_effort seam (logged, never raised)."""

    def test_worker_shutdown_failure_degrades_and_logs(self, caplog):
        class FakeIter:
            def _shutdown_workers(self):
                raise RuntimeError("worker pipe broken")

        class FakeLoader:
            num_workers = 2
            _iterator = FakeIter()

        with caplog.at_level(logging.DEBUG, logger="mimarsinan.best_effort"):
            shutdown_data_loader(FakeLoader())
        assert any("shutdown" in r.getMessage() for r in caplog.records)

    def test_unregister_with_exploding_iterator_degrades_and_logs(self, caplog):
        class Weird:
            @property
            def _workers(self):
                raise RuntimeError("boom")

        with caplog.at_level(logging.DEBUG, logger="mimarsinan.best_effort"):
            _unregister_dataloader_atexit_handlers(Weird())
        assert any("atexit" in r.getMessage() for r in caplog.records)

    def test_resource_snapshot_reads_env_at_call_time(self, monkeypatch, capsys):
        import mimarsinan.data_handling.data_loader_factory as dlf_mod

        monkeypatch.delenv("MIMARSINAN_RESOURCE_DEBUG", raising=False)
        dlf_mod._resource_snapshot("disabled-tag")
        assert "disabled-tag" not in capsys.readouterr().err
        monkeypatch.setenv("MIMARSINAN_RESOURCE_DEBUG", "1")
        dlf_mod._resource_snapshot("enabled-tag")
        assert "enabled-tag" in capsys.readouterr().err

    def test_resource_snapshot_failure_degrades_and_logs(self, monkeypatch, caplog):
        import mimarsinan.data_handling.data_loader_factory as dlf_mod

        def exploding_children():
            raise RuntimeError("mp broken")

        monkeypatch.setenv("MIMARSINAN_RESOURCE_DEBUG", "1")
        monkeypatch.setattr(dlf_mod._mp, "active_children", exploding_children)
        with caplog.at_level(logging.DEBUG, logger="mimarsinan.best_effort"):
            dlf_mod._resource_snapshot("test-tag")
        assert any("snapshot" in r.getMessage() for r in caplog.records)
