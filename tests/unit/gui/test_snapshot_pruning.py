"""Unit tests for snapshot_pruning_layers (Pruning Adaptation step GUI data).

Sharpened contract: every skip path (missing masks, mask-length mismatch,
bare-``nn.Linear`` fallback, no layers at all) surfaces a structured
``{layer, reason}`` entry in the summary's ``skipped`` list instead of a
silent ``continue`` — the tab renders these as visible diagnostics.
"""

import pytest
import torch

from mimarsinan.gui.snapshot.builders import (
    RESOURCE_KIND_PRUNING_LAYER_HEATMAP,
    RESOURCE_KIND_PRUNING_MASK_MAP,
    snapshot_pruning_layers,
)
from mimarsinan.gui.snapshot.util.constants import RESOURCE_KIND_HEATMAP_COLORBAR


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
    def test_returns_layers_and_skipped_keys(self):
        model = _ModelWithPerceptrons([])
        out, descriptors = snapshot_pruning_layers(model)
        assert out["layers"] == []
        assert descriptors == []
        # No extractable layers is itself a surfaced diagnostic, not a bare empty tab.
        assert len(out["skipped"]) == 1
        assert "layer" in out["skipped"][0] and "reason" in out["skipped"][0]

    def test_missing_masks_surface_as_skipped_diagnostic(self):
        """Sharpened from the old silent-empty pin: a maskless layer must
        appear in ``skipped`` — restoring the silent ``continue`` fails here."""
        layer = torch.nn.Linear(4, 3)
        layer.weight.data = torch.randn(3, 4) * 0.1
        p = type("_P", (), {"layer": layer, "name": "no_masks"})()
        model = _ModelWithPerceptrons([p])
        out, descriptors = snapshot_pruning_layers(model)
        assert out["layers"] == []
        assert descriptors == []
        assert [e["layer"] for e in out["skipped"]] == ["no_masks"]
        assert "mask" in out["skipped"][0]["reason"]

    def test_mask_length_mismatch_surfaces_as_skipped_diagnostic(self):
        """Sharpened from the old silent-empty pin: a length mismatch names
        the layer and the mismatching lengths."""
        layer = torch.nn.Linear(4, 3)
        layer.weight.data = torch.randn(3, 4) * 0.1
        layer.register_buffer("prune_row_mask", torch.zeros(3, dtype=torch.bool))
        layer.register_buffer("prune_col_mask", torch.zeros(2, dtype=torch.bool))  # wrong length
        p = type("_P", (), {"layer": layer, "name": "bad"})()
        model = _ModelWithPerceptrons([p])
        out, descriptors = snapshot_pruning_layers(model)
        assert out["layers"] == []
        assert descriptors == []
        assert [e["layer"] for e in out["skipped"]] == ["bad"]
        reason = out["skipped"][0]["reason"]
        assert "mismatch" in reason and "2" in reason and "4" in reason

    def test_bare_linear_fallback_is_surfaced_not_silently_empty(self):
        """The bare-``nn.Linear`` fallback synthesizes wrappers that never carry
        masks; the summary must say so instead of rendering an empty tab."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        out, descriptors = snapshot_pruning_layers(model)
        assert out["layers"] == []
        assert descriptors == []
        reasons = " | ".join(e["reason"] for e in out["skipped"])
        assert "nn.Linear fallback" in reasons
        # The maskless synthesized layer is also individually surfaced.
        assert any(e["layer"] == "linear_0" for e in out["skipped"])

    def test_includes_layers_with_matching_masks(self):
        p0 = _make_perceptron(3, 4, "layer_0", pruned_rows=1, pruned_cols=2)
        p1 = _make_perceptron(2, 3, "layer_1", pruned_rows=0, pruned_cols=1)
        model = _ModelWithPerceptrons([p0, p1])
        out, descriptors = snapshot_pruning_layers(model)
        assert len(out["layers"]) == 2
        assert out["skipped"] == []
        for i, L in enumerate(out["layers"]):
            assert L["layer_index"] == i
            assert "layer_name" in L
            assert len(L["shape"]) == 2
            # Summary must not embed heatmap bytes.
            assert "heatmap_image" not in L
            assert L.get("has_heatmap") is True
            assert L["heatmap_resource"]["kind"] == RESOURCE_KIND_PRUNING_LAYER_HEATMAP
            assert L["heatmap_resource"]["rid"] == f"layer/{i}"
            assert L["mask_map_resource"]["kind"] == RESOURCE_KIND_PRUNING_MASK_MAP
            assert L["mask_map_resource"]["rid"] == f"layer/{i}"
        assert out["layers"][0]["shape"] == [3, 4]
        assert out["layers"][0]["pruned_rows"] == 1
        assert out["layers"][0]["pruned_cols"] == 2
        assert out["layers"][1]["shape"] == [2, 3]
        assert out["layers"][1]["pruned_rows"] == 0
        assert out["layers"][1]["pruned_cols"] == 1

        # One heatmap + one mask-map descriptor per layer (rids matching the
        # summary) plus exactly ONE family colorbar for the weight heatmaps.
        assert len(descriptors) == 5
        by_kind: dict = {}
        for d in descriptors:
            by_kind.setdefault(d.kind, set()).add(d.rid)
        assert by_kind[RESOURCE_KIND_PRUNING_LAYER_HEATMAP] == {"layer/0", "layer/1"}
        assert by_kind[RESOURCE_KIND_PRUNING_MASK_MAP] == {"layer/0", "layer/1"}
        assert len(by_kind[RESOURCE_KIND_HEATMAP_COLORBAR]) == 1
        assert all(d.media_type == "image/png" for d in descriptors)

        # Weight heatmaps share ONE family scale, reported in the summary;
        # mask maps stay on their own categorical +-1 scale (unstamped).
        layer_descriptors = [
            d for d in descriptors if d.kind == RESOURCE_KIND_PRUNING_LAYER_HEATMAP
        ]
        mask_maps = [d for d in descriptors if d.kind == RESOURCE_KIND_PRUNING_MASK_MAP]
        assert len({d.source.scale for d in layer_descriptors}) == 1
        assert out["heatmap_scale"]["vmax"] == layer_descriptors[0].source.scale
        assert all(d.source.scale is None for d in mask_maps)
        # Producers lazily render PNG bytes.
        for d in (layer_descriptors[0], mask_maps[0]):
            png_bytes = d.producer()
            assert isinstance(png_bytes, bytes)
            assert png_bytes.startswith(b"\x89PNG")

    def test_pre_post_dimensions_and_achieved_sparsity(self):
        """Committed weights stay FULL-width on the model (zeroed rows), so
        pre = weight.shape, post = kept counts, sparsity = pruned/len."""
        p = _make_perceptron(4, 8, "fc", pruned_rows=1, pruned_cols=2)
        model = _ModelWithPerceptrons([p])
        out, _ = snapshot_pruning_layers(model)
        L = out["layers"][0]
        assert L["pre_neurons"] == 4
        assert L["pre_axons"] == 8
        assert L["post_neurons"] == 3
        assert L["post_axons"] == 6
        assert L["achieved_sparsity_neurons"] == pytest.approx(1 / 4)
        assert L["achieved_sparsity_axons"] == pytest.approx(2 / 8)

    def test_configured_fraction_is_plumbed_into_the_summary(self):
        model = _ModelWithPerceptrons([_make_perceptron(3, 4, "fc")])
        out, _ = snapshot_pruning_layers(model, configured_fraction=0.25)
        assert out["configured_fraction"] == 0.25
        out_default, _ = snapshot_pruning_layers(model)
        assert out_default["configured_fraction"] is None

    def test_mask_map_matrix_encodes_kept_vs_pruned(self):
        """The mask map is the outer product of the keep-masks: +1 kept,
        -1 pruned, same shape as the weight, with the red-line masks attached."""
        p = _make_perceptron(3, 4, "fc", pruned_rows=1, pruned_cols=2)
        model = _ModelWithPerceptrons([p])
        _, descriptors = snapshot_pruning_layers(model)
        mask_maps = [d for d in descriptors if d.kind == RESOURCE_KIND_PRUNING_MASK_MAP]
        assert len(mask_maps) == 1
        source = mask_maps[0].source
        matrix = source.matrix
        assert matrix.shape == (3, 4)
        assert (matrix[0, :] == -1.0).all()  # pruned row
        assert (matrix[:, 0] == -1.0).all()  # pruned col
        assert (matrix[:, 1] == -1.0).all()  # pruned col
        assert (matrix[1:, 2:] == 1.0).all()  # surviving submatrix
        assert source.pruned_row_mask == [True, False, False]
        assert source.pruned_col_mask == [True, True, False, False]
