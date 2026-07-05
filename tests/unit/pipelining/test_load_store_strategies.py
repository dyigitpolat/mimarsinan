"""Tests for individual load/store strategies."""

import copy
import logging

import pytest
import torch
import torch.nn as nn

from conftest import make_tiny_supermodel
from mimarsinan.pipelining.cache.load_store_strategies import (
    BasicLoadStoreStrategy,
    TorchModelLoadStoreStrategy,
    PickleLoadStoreStrategy,
)
from mimarsinan.tuning.tuners.pruning.pruning_tuner_enforce import (
    enforce_pruning_persistently,
    register_prune_buffers,
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


def _pruned_supermodel():
    """A tiny supermodel with committed prune masks + enforcement hooks."""
    torch.manual_seed(0)
    model = make_tiny_supermodel()
    perceptrons = model.get_perceptrons()
    row_masks, col_masks = [], []
    for p in perceptrons:
        out_f, in_f = p.layer.weight.shape
        row_keep = torch.ones(out_f, dtype=torch.bool)
        row_keep[0] = False
        col_keep = torch.ones(in_f, dtype=torch.bool)
        row_masks.append(row_keep)
        col_masks.append(col_keep)
    register_prune_buffers(perceptrons, row_masks, col_masks)
    enforce_pruning_persistently(perceptrons, row_masks, col_masks)
    return model


def _dirty_pruned_rows(model):
    """Simulate pruned-row regrowth in raw params (no forward fires afterwards)."""
    with torch.no_grad():
        for p in model.get_perceptrons():
            p.layer.weight[p.layer.prune_mask] = 3.0
            if p.layer.bias is not None:
                p.layer.bias[p.layer.prune_bias_mask] = 3.0


def _accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return float(model(x).argmax(dim=1).eq(y).float().mean())


class TestTorchModelPrunedCacheRoundTrip:
    """W-CAL prune-parity at the cache boundary (theory §5g-v incidental (i)).

    Old pruned artifacts reloaded 4.8-11.2 pp below their shipped live metrics:
    raw weights carried fewer zero rows than the persisted masks enforce. The
    store boundary must COMMIT the masks into raw params (mask.param == param
    holds in the artifact) and the load boundary must fail-loud VERIFY, mirroring
    the mapping-time verify in SoftCoreMappingStep.
    """

    def test_store_commits_regrown_rows_and_roundtrip_preserves_metric(self, tmp_path):
        model = _pruned_supermodel()
        x = torch.randn(8, 1, 8, 8)
        y = torch.randint(0, 4, (8,))
        reference = copy.deepcopy(model)
        reference.eval()
        with torch.no_grad():
            y_expected = reference(x).clone()
        expected_acc = _accuracy(reference, x, y)

        _dirty_pruned_rows(model)
        s = TorchModelLoadStoreStrategy("pruned_model")
        s.store(str(tmp_path), model)

        raw, _device = torch.load(
            str(tmp_path / "pruned_model.pt"),
            map_location="cpu", weights_only=False,
        )
        for i, p in enumerate(raw.get_perceptrons()):
            masked = p.layer.weight.detach()[p.layer.prune_mask]
            assert torch.equal(masked, torch.zeros_like(masked)), (
                f"stored artifact perceptron {i}: mask.param == param must hold "
                "in the raw file, before any forward fires the hooks"
            )

        loaded = s.load(str(tmp_path))
        loaded.eval()
        with torch.no_grad():
            y_loaded = loaded(x)
        assert torch.equal(y_loaded, y_expected), (
            "reloaded model must reproduce the live (mask-enforced) behavior"
        )
        assert _accuracy(loaded, x, y) == expected_acc

    def test_store_commits_on_the_live_object_too(self, tmp_path):
        model = _pruned_supermodel()
        _dirty_pruned_rows(model)
        TorchModelLoadStoreStrategy("live_commit").store(str(tmp_path), model)
        for p in model.get_perceptrons():
            masked = p.layer.weight.detach()[p.layer.prune_mask]
            assert torch.equal(masked, torch.zeros_like(masked))

    def test_load_fails_loud_on_poisoned_artifact(self, tmp_path):
        model = _pruned_supermodel()
        _dirty_pruned_rows(model)
        # Bypass the store-side commit: a pre-contract (poisoned) artifact.
        torch.save((model, torch.device("cpu")), str(tmp_path / "poisoned.pt"))
        with pytest.raises(RuntimeError, match="prun"):
            TorchModelLoadStoreStrategy("poisoned").load(str(tmp_path))

    def test_maskless_model_roundtrip_untouched(self, tmp_path):
        torch.manual_seed(0)
        model = make_tiny_supermodel()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        s = TorchModelLoadStoreStrategy("maskless")
        s.store(str(tmp_path), model)
        loaded = s.load(str(tmp_path))
        after = loaded.state_dict()
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key])


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
