"""Unit tests for snapshot_pruning_layers (Pruning Adaptation step GUI data)."""

import pytest
import torch

from mimarsinan.gui.snapshot.builders import snapshot_pruning_layers


def _make_perceptron(out_f: int, in_f: int, name: str, pruned_rows: int = 0, pruned_cols: int = 0):
    """Build a minimal perceptron-like object with layer.weight and pruning masks."""
    layer = torch.nn.Linear(in_f, out_f)
    layer.weight.data = torch.randn(out_f, in_f) * 0.1
    # Model convention: True = pruned
    row_mask = torch.zeros(out_f, dtype=torch.bool)
    row_mask[:pruned_rows] = True
    col_mask = torch.zeros(in_f, dtype=torch.bool)
    col_mask[:pruned_cols] = True
    layer.register_buffer("prune_row_mask", row_mask)
    layer.register_buffer("prune_col_mask", col_mask)
    wrapper = type("_P", (), {"layer": layer, "name": name})()
    return wrapper


class _ModelWithPerceptrons:
    def __init__(self, perceptrons):
        self._perceptrons = perceptrons

    def get_perceptrons(self):
        return self._perceptrons


class TestSnapshotPruningLayers:
    def test_returns_layers_key_with_list(self):
        model = _ModelWithPerceptrons([])
        out = snapshot_pruning_layers(model)
        assert "layers" in out
        assert out["layers"] == []

    def test_skips_perceptrons_without_masks(self):
        layer = torch.nn.Linear(4, 3)
        layer.weight.data = torch.randn(3, 4) * 0.1
        p = type("_P", (), {"layer": layer, "name": "no_masks"})()
        model = _ModelWithPerceptrons([p])
        out = snapshot_pruning_layers(model)
        assert len(out["layers"]) == 0

    def test_includes_layers_with_matching_masks(self):
        p0 = _make_perceptron(3, 4, "layer_0", pruned_rows=1, pruned_cols=2)
        p1 = _make_perceptron(2, 3, "layer_1", pruned_rows=0, pruned_cols=1)
        model = _ModelWithPerceptrons([p0, p1])
        out = snapshot_pruning_layers(model)
        assert len(out["layers"]) == 2
        for i, L in enumerate(out["layers"]):
            assert "layer_index" in L
            assert L["layer_index"] == i
            assert "layer_name" in L
            assert "shape" in L
            assert len(L["shape"]) == 2
            assert "pruned_rows" in L
            assert "pruned_cols" in L
            assert "heatmap_image" in L
            assert L["heatmap_image"].startswith("data:image/png;base64,")
        assert out["layers"][0]["shape"] == [3, 4]
        assert out["layers"][0]["pruned_rows"] == 1
        assert out["layers"][0]["pruned_cols"] == 2
        assert out["layers"][1]["shape"] == [2, 3]
        assert out["layers"][1]["pruned_rows"] == 0
        assert out["layers"][1]["pruned_cols"] == 1

    def test_skips_layer_when_mask_length_mismatch(self):
        layer = torch.nn.Linear(4, 3)
        layer.weight.data = torch.randn(3, 4) * 0.1
        layer.register_buffer("prune_row_mask", torch.zeros(3, dtype=torch.bool))
        layer.register_buffer("prune_col_mask", torch.zeros(2, dtype=torch.bool))  # wrong length
        p = type("_P", (), {"layer": layer, "name": "bad"})()
        model = _ModelWithPerceptrons([p])
        out = snapshot_pruning_layers(model)
        assert len(out["layers"]) == 0
