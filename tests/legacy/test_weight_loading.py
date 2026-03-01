"""Tests for weight loading strategies and the WeightPreloadingStep."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mimarsinan.model_training.weight_loading import (
    TorchvisionWeightStrategy,
    CheckpointWeightStrategy,
    URLWeightStrategy,
    resolve_weight_strategy,
)


class SimpleMLP(nn.Module):
    def __init__(self, in_f=16, hidden=32, out_f=10):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.fc2 = nn.Linear(hidden, out_f)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ── TorchvisionWeightStrategy ───────────────────────────────────────────────

class TestTorchvisionWeightStrategy:
    def test_load_from_factory(self):
        target = SimpleMLP(16, 32, 10)
        source_weights = SimpleMLP(16, 32, 10).state_dict()

        def factory():
            m = SimpleMLP(16, 32, 10)
            m.load_state_dict(source_weights)
            return m

        strategy = TorchvisionWeightStrategy(factory)
        model, info = strategy.load(target)

        assert info["matched"] > 0
        assert len(info["missing_keys"]) == 0
        assert len(info["unexpected_keys"]) == 0

        for k in source_weights:
            assert torch.allclose(model.state_dict()[k], source_weights[k])

    def test_load_with_mismatched_head(self):
        target = SimpleMLP(16, 32, 5)

        def factory():
            return SimpleMLP(16, 32, 10)

        strategy = TorchvisionWeightStrategy(factory)
        model, info = strategy.load(target)

        assert len(info["missing_keys"]) > 0 or len(info["unexpected_keys"]) > 0
        assert info["matched"] > 0


# ── CheckpointWeightStrategy ────────────────────────────────────────────────

class TestCheckpointWeightStrategy:
    def test_load_from_state_dict(self, tmp_path):
        source = SimpleMLP(16, 32, 10)
        ckpt_path = tmp_path / "model.pt"
        torch.save(source.state_dict(), ckpt_path)

        target = SimpleMLP(16, 32, 10)
        strategy = CheckpointWeightStrategy(ckpt_path)
        model, info = strategy.load(target)

        assert info["matched"] > 0
        for k in source.state_dict():
            assert torch.allclose(model.state_dict()[k], source.state_dict()[k])

    def test_load_from_wrapped_dict(self, tmp_path):
        source = SimpleMLP(16, 32, 10)
        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save({"model_state_dict": source.state_dict(), "epoch": 5}, ckpt_path)

        target = SimpleMLP(16, 32, 10)
        strategy = CheckpointWeightStrategy(ckpt_path)
        model, info = strategy.load(target)

        assert info["matched"] > 0

    def test_missing_file_raises(self, tmp_path):
        strategy = CheckpointWeightStrategy(tmp_path / "nonexistent.pt")
        with pytest.raises(FileNotFoundError):
            strategy.load(SimpleMLP())


# ── resolve_weight_strategy ──────────────────────────────────────────────────

class TestResolveWeightStrategy:
    def test_none_returns_none(self):
        assert resolve_weight_strategy(None) is None
        assert resolve_weight_strategy("") is None

    def test_torchvision_requires_builder(self):
        with pytest.raises(ValueError, match="get_pretrained_factory"):
            resolve_weight_strategy("torchvision", model_builder=None)

    def test_torchvision_with_builder(self):
        class FakeBuilder:
            def get_pretrained_factory(self):
                return lambda: SimpleMLP()

        strategy = resolve_weight_strategy("torchvision", model_builder=FakeBuilder())
        assert isinstance(strategy, TorchvisionWeightStrategy)

    def test_file_path(self, tmp_path):
        p = tmp_path / "weights.pt"
        p.touch()
        strategy = resolve_weight_strategy(str(p))
        assert isinstance(strategy, CheckpointWeightStrategy)

    def test_url(self):
        strategy = resolve_weight_strategy("https://example.com/model.pt")
        assert isinstance(strategy, URLWeightStrategy)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_weight_strategy("some_random_string")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
