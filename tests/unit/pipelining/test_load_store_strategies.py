"""Tests for individual load/store strategies."""

import logging

import pytest
import torch
import torch.nn as nn

from mimarsinan.pipelining.cache.load_store_strategies import (
    BasicLoadStoreStrategy,
    TorchModelLoadStoreStrategy,
    PickleLoadStoreStrategy,
)


class TestBasicLoadStoreStrategy:
    def test_scalar(self, tmp_path):
        s = BasicLoadStoreStrategy("val")
        s.store(str(tmp_path), 42)
        assert s.load(str(tmp_path)) == 42

    def test_dict(self, tmp_path):
        s = BasicLoadStoreStrategy("cfg")
        data = {"lr": 0.001, "epochs": 10, "nested": {"a": 1}}
        s.store(str(tmp_path), data)
        loaded = s.load(str(tmp_path))
        assert loaded == data

    def test_list(self, tmp_path):
        s = BasicLoadStoreStrategy("lst")
        s.store(str(tmp_path), [1.0, 2.0, 3.0])
        assert s.load(str(tmp_path)) == [1.0, 2.0, 3.0]

    def test_empty_dict(self, tmp_path):
        s = BasicLoadStoreStrategy("empty")
        s.store(str(tmp_path), {})
        assert s.load(str(tmp_path)) == {}

    def test_string(self, tmp_path):
        s = BasicLoadStoreStrategy("str")
        s.store(str(tmp_path), "hello")
        assert s.load(str(tmp_path)) == "hello"


class TestTorchModelLoadStoreStrategy:
    def test_linear(self, tmp_path):
        model = nn.Linear(8, 4)
        s = TorchModelLoadStoreStrategy("model")
        s.store(str(tmp_path), model)
        loaded = s.load(str(tmp_path))
        assert isinstance(loaded, nn.Linear)
        assert torch.allclose(model.weight.data, loaded.weight.data)

    def test_sequential(self, tmp_path):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        s = TorchModelLoadStoreStrategy("seq")
        s.store(str(tmp_path), model)
        loaded = s.load(str(tmp_path))
        assert isinstance(loaded, nn.Sequential)
        x = torch.randn(1, 4)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x))

    def test_preserves_cpu_device(self, tmp_path):
        model = nn.Linear(2, 2)
        s = TorchModelLoadStoreStrategy("m")
        s.store(str(tmp_path), model)
        loaded = s.load(str(tmp_path))
        assert next(loaded.parameters()).device == torch.device("cpu")


class _RestoreFailModel(nn.Module):
    """Model whose ``.to()`` raises, simulating a vanished CUDA device after save."""

    def __init__(self, exc_type):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self._exc_type = exc_type

    def to(self, *args, **kwargs):
        raise self._exc_type("restore failed")


class TestTorchModelStoreDeviceRestore:
    def test_runtime_error_falls_back_to_cpu_with_warning(self, tmp_path, caplog):
        model = _RestoreFailModel(RuntimeError)
        s = TorchModelLoadStoreStrategy("m_restore")
        with caplog.at_level(
            logging.WARNING,
            logger="mimarsinan.pipelining.cache.load_store_strategies",
        ):
            s.store(str(tmp_path), model)
        assert (tmp_path / "m_restore.pt").exists()
        assert any("leaving on CPU" in r.getMessage() for r in caplog.records)
        assert next(model.parameters()).device == torch.device("cpu")

    def test_unexpected_error_propagates(self, tmp_path):
        model = _RestoreFailModel(ValueError)
        s = TorchModelLoadStoreStrategy("m_restore2")
        with pytest.raises(ValueError, match="restore failed"):
            s.store(str(tmp_path), model)


class TestPickleLoadStoreStrategy:
    def test_dict_with_tensors(self, tmp_path):
        obj = {"a": torch.tensor([1.0, 2.0]), "b": "text"}
        s = PickleLoadStoreStrategy("blob")
        s.store(str(tmp_path), obj)
        loaded = s.load(str(tmp_path))
        assert torch.allclose(loaded["a"], torch.tensor([1.0, 2.0]))
        assert loaded["b"] == "text"

    def test_nested_list(self, tmp_path):
        obj = [[1, 2], [3, 4]]
        s = PickleLoadStoreStrategy("nested")
        s.store(str(tmp_path), obj)
        assert s.load(str(tmp_path)) == [[1, 2], [3, 4]]

    def test_none(self, tmp_path):
        s = PickleLoadStoreStrategy("none")
        s.store(str(tmp_path), None)
        assert s.load(str(tmp_path)) is None

    def test_set_object(self, tmp_path):
        s = PickleLoadStoreStrategy("myset")
        s.store(str(tmp_path), {1, 2, 3})
        loaded = s.load(str(tmp_path))
        assert loaded == {1, 2, 3}
