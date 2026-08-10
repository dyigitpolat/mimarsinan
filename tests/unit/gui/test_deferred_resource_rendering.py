"""Snapshot-persist renders CHEAP UI-resolution artifacts under BOTH policies.

The matplotlib-era contract was "a headless run renders nothing" because a
1024px pyplot render backlog once pinned the process for minutes after its
last step. The pure-numpy renderer made the UI-resolution artifact cheap, so
the sharpened contract is: EVERY run persists each resource's SOURCE (the
full-resolution zoom feeds off it on demand) AND its UI-resolution artifact
(the browser's first attach is a plain file read, never a render storm) --
and NOTHING at persist time may pay for a full-resolution render.
"""

from __future__ import annotations

import gc
import json
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mimarsinan.common.env import GUI_RESOURCE_RENDER_VALUES
from mimarsinan.gui.handle import GUIHandle
from mimarsinan.gui.resources import (
    HeatmapSource,
    JsonSource,
    ResourceDescriptor,
    ResourceRenderPolicy,
    encode_resource_payload,
    resolve_resource_render_policy,
)
from mimarsinan.gui.runtime.collector import DataCollector
from mimarsinan.gui.runtime.persistence import (
    load_resource_from_disk,
    load_resource_source,
    resource_source_disk_path,
    save_resource_source,
)
from mimarsinan.gui.runtime.persistence.resource_paths import resource_disk_path
from mimarsinan.gui.snapshot import snapshot_pruning_layers

STEP = "Pruning Adaptation"


class _ModelWithPerceptrons:
    def __init__(self, perceptrons):
        self._perceptrons = perceptrons

    def get_perceptrons(self):
        return self._perceptrons


def _pruning_descriptors():
    """Real descriptors from the real snapshot builder (one PNG heatmap)."""
    layer = torch.nn.Linear(4, 3)
    layer.weight.data = torch.arange(12, dtype=torch.float32).reshape(3, 4) * 0.1
    layer.register_buffer("prune_row_mask", torch.tensor([True, False, False]))
    layer.register_buffer("prune_col_mask", torch.tensor([False, True, False, False]))
    perceptron = SimpleNamespace(layer=layer, name="fc0")
    _, descriptors = snapshot_pruning_layers(_ModelWithPerceptrons([perceptron]))
    assert descriptors, "the pruning snapshot must register a heatmap descriptor"
    return descriptors


def _handle(working_dir, descriptors, policy, monkeypatch):
    monkeypatch.setattr(
        "mimarsinan.gui.handle.build_step_snapshot",
        MagicMock(return_value=({"step_name": STEP}, {}, descriptors)),
    )
    collector = DataCollector()
    pipeline = SimpleNamespace(
        get_target_metric=MagicMock(return_value=None),
        working_directory=working_dir,
    )
    gui = GUIHandle(pipeline, collector, capture_stdio=False, render_policy=policy)
    collector.step_started(STEP)
    return gui


class TestPersistTimeRendersOnlyUiResolution:
    """THE defect (sharpened): the exit drain once paid a full-resolution
    matplotlib backlog inline. Persist time may now render ONLY the cheap
    UI-resolution artifact -- never the full-resolution variant."""

    def test_deferred_step_end_renders_at_ui_resolution_only(self, tmp_path, monkeypatch):
        from mimarsinan.gui.heatmap_renderer import (
            DEFAULT_TARGET_LONG_SIDE,
            render_heatmap_png_bytes,
        )

        seen_targets: list[int] = []

        def _capturing_renderer(*args, **kwargs):
            seen_targets.append(
                kwargs.get("target_long_side", DEFAULT_TARGET_LONG_SIDE)
            )
            return render_heatmap_png_bytes(*args, **kwargs)

        monkeypatch.setattr(
            "mimarsinan.gui.heatmap_renderer.render_heatmap_png_bytes",
            _capturing_renderer,
        )
        gui = _handle(
            str(tmp_path), _pruning_descriptors(),
            ResourceRenderPolicy.DEFERRED, monkeypatch,
        )
        gui.on_step_end(STEP, SimpleNamespace())
        assert gui.wait_snapshots_idle(timeout=10.0)

        assert seen_targets, "the deferred persist must render the UI artifact"
        assert all(t <= DEFAULT_TARGET_LONG_SIDE for t in seen_targets), (
            "persist time paid for a beyond-UI-resolution render"
        )

    def test_deferred_step_end_persists_both_source_and_ui_artifact(self, tmp_path, monkeypatch):
        descriptors = _pruning_descriptors()
        gui = _handle(
            str(tmp_path), descriptors, ResourceRenderPolicy.DEFERRED, monkeypatch,
        )

        gui.on_step_end(STEP, SimpleNamespace())
        assert gui.wait_snapshots_idle(timeout=10.0)

        desc = descriptors[0]
        assert resource_source_disk_path(
            str(tmp_path), STEP, desc.kind, desc.rid,
        ).is_file()
        rendered = resource_disk_path(
            str(tmp_path), STEP, desc.kind, desc.rid, desc.media_type,
        )
        assert rendered.is_file(), "DEFERRED must pre-render the UI-res artifact"
        assert rendered.read_bytes() == desc.source.render()


class TestAttachedMonitorStillRendersEagerly:
    def test_eager_policy_renders_and_writes_the_resource_and_source(self, tmp_path, monkeypatch):
        descriptors = _pruning_descriptors()
        gui = _handle(
            str(tmp_path), descriptors, ResourceRenderPolicy.EAGER, monkeypatch,
        )

        gui.on_step_end(STEP, SimpleNamespace())
        assert gui.wait_snapshots_idle(timeout=10.0)

        desc = descriptors[0]
        rendered = resource_disk_path(
            str(tmp_path), STEP, desc.kind, desc.rid, desc.media_type,
        )
        assert rendered.is_file()
        assert rendered.read_bytes().startswith(b"\x89PNG")
        # The source is persisted under EAGER too: the on-demand
        # full-resolution variant feeds off it for historical runs.
        assert resource_source_disk_path(
            str(tmp_path), STEP, desc.kind, desc.rid,
        ).is_file()


class TestNothingIsLost:
    """A persisted source must reproduce the resource the eager path renders."""

    def test_deferred_run_reproduces_the_eager_bytes(self, tmp_path, monkeypatch):
        eager_dir = tmp_path / "eager"
        deferred_dir = tmp_path / "deferred"
        eager_dir.mkdir()
        deferred_dir.mkdir()

        eager_descriptors = _pruning_descriptors()
        gui = _handle(
            str(eager_dir), eager_descriptors, ResourceRenderPolicy.EAGER, monkeypatch,
        )
        gui.on_step_end(STEP, SimpleNamespace())
        assert gui.wait_snapshots_idle(timeout=10.0)

        deferred_descriptors = _pruning_descriptors()
        gui = _handle(
            str(deferred_dir), deferred_descriptors,
            ResourceRenderPolicy.DEFERRED, monkeypatch,
        )
        gui.on_step_end(STEP, SimpleNamespace())
        assert gui.wait_snapshots_idle(timeout=5.0)

        desc = eager_descriptors[0]
        eager_bytes = resource_disk_path(
            str(eager_dir), STEP, desc.kind, desc.rid, desc.media_type,
        ).read_bytes()
        reproduced = load_resource_from_disk(
            str(deferred_dir), STEP, desc.kind, desc.rid, media_type=desc.media_type,
        )
        assert reproduced == eager_bytes

    def test_attach_caches_the_rendered_bytes_for_the_next_read(self, tmp_path):
        source = HeatmapSource(np.arange(12, dtype=np.float64).reshape(3, 4))
        save_resource_source(str(tmp_path), STEP, "pruning_layer_heatmap", "layer/0", source)
        cached = resource_disk_path(
            str(tmp_path), STEP, "pruning_layer_heatmap", "layer/0", "image/png",
        )
        assert not cached.exists()

        first = load_resource_from_disk(
            str(tmp_path), STEP, "pruning_layer_heatmap", "layer/0", media_type="image/png",
        )
        assert first is not None and first.startswith(b"\x89PNG")
        assert cached.is_file() and cached.read_bytes() == first

        cached.write_bytes(b"\x89PNG-cached")
        assert load_resource_from_disk(
            str(tmp_path), STEP, "pruning_layer_heatmap", "layer/0", media_type="image/png",
        ) == b"\x89PNG-cached"

    def test_missing_source_and_missing_render_is_a_miss(self, tmp_path):
        assert load_resource_from_disk(
            str(tmp_path), STEP, "connectivity", "seg/0/core/0",
            media_type="application/json",
        ) is None


class TestResourceSourceRoundTrip:
    @pytest.mark.parametrize("dtype", ["float64", "float32", "int8", "uint8", "bool", "int64"])
    def test_dense_and_sparse_arrays_round_trip_bit_exactly(self, tmp_path, dtype):
        rng = np.random.default_rng(0)
        dense = (rng.random((16, 24)) * 100).astype(dtype)
        sparse = np.zeros((64, 64), dtype=dtype)
        sparse[:4, :4] = dense[:4, :4]
        for rid, matrix in (("dense", dense), ("sparse", sparse)):
            save_resource_source(
                str(tmp_path), STEP, "ir_core_heatmap", rid, HeatmapSource(matrix),
            )
            loaded = load_resource_source(str(tmp_path), STEP, "ir_core_heatmap", rid)
            assert loaded is not None
            assert loaded.matrix.dtype == matrix.dtype
            assert np.array_equal(loaded.matrix, matrix)

    def test_negative_zero_and_nan_survive_the_sparse_encoding(self, tmp_path):
        matrix = np.zeros((32, 32), dtype=np.float64)
        matrix[0, 0] = -0.0
        matrix[0, 1] = np.nan
        matrix[0, 2] = 1.5
        save_resource_source(str(tmp_path), STEP, "ir_core_heatmap", "z", HeatmapSource(matrix))
        loaded = load_resource_source(str(tmp_path), STEP, "ir_core_heatmap", "z")
        assert loaded is not None
        assert loaded.matrix.tobytes() == matrix.tobytes()

    def test_masks_and_render_survive_the_round_trip(self, tmp_path):
        matrix = np.arange(20, dtype=np.float64).reshape(4, 5)
        source = HeatmapSource(
            matrix, pruned_row_mask=[True, False, False, True],
            pruned_col_mask=[False] * 5,
        )
        save_resource_source(str(tmp_path), STEP, "ir_core_pre_pruning", "core/1", source)
        loaded = load_resource_source(str(tmp_path), STEP, "ir_core_pre_pruning", "core/1")
        assert loaded is not None
        assert loaded.pruned_row_mask == [True, False, False, True]
        assert loaded.render() == source.render()

    def test_json_source_round_trips_and_renders_identically(self, tmp_path):
        payload = [{"src_core": 0, "dst_core": 1, "length": 4, "kind": "core"}]
        save_resource_source(str(tmp_path), STEP, "connectivity", "seg/0/core/0", JsonSource(payload))
        loaded = load_resource_source(str(tmp_path), STEP, "connectivity", "seg/0/core/0")
        assert loaded is not None
        assert loaded.render() == payload
        assert encode_resource_payload(loaded.render(), "application/json") == json.dumps(payload).encode()

    def test_unreadable_source_file_is_a_miss_not_a_crash(self, tmp_path):
        path = resource_source_disk_path(str(tmp_path), STEP, "ir_core_heatmap", "bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a resource source")
        assert load_resource_source(str(tmp_path), STEP, "ir_core_heatmap", "bad") is None


class TestSourcesOwnHostMemoryOnly:
    """No producer may keep a device buffer alive past the step that made it."""

    def test_torch_tensor_is_copied_out_and_released(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        source = HeatmapSource(tensor)
        assert isinstance(source.matrix, np.ndarray)
        assert np.array_equal(source.matrix, tensor.numpy())
        assert not np.shares_memory(source.matrix, tensor.numpy())
        assert source.matrix.base is None

        alive = weakref.ref(tensor)
        del tensor
        gc.collect()
        assert alive() is None, "the source is still holding the tensor it was built from"
        assert source.matrix.shape == (3, 4)

    def test_device_tensor_that_cannot_convert_fails_loud(self):
        class _DeviceTensor:
            def detach(self):
                return self

            def cpu(self):
                raise RuntimeError("no host copy available")

            def numpy(self):
                raise TypeError("cannot convert a device tensor to numpy")

        with pytest.raises(RuntimeError):
            HeatmapSource(_DeviceTensor())

    def test_non_numeric_payload_is_rejected(self):
        with pytest.raises(TypeError):
            HeatmapSource([{"not": "an array"}])


class TestRenderPolicyIsDeclared:
    def test_env_vocabulary_matches_the_policy_enum(self):
        """The env SSOT parses the names; the enum is what the code branches on."""
        assert set(GUI_RESOURCE_RENDER_VALUES) == {p.value for p in ResourceRenderPolicy}

    def test_declared_policy_is_used_when_no_override(self, monkeypatch):
        monkeypatch.delenv("MIMARSINAN_GUI_RESOURCE_RENDER", raising=False)
        for declared in ResourceRenderPolicy:
            assert resolve_resource_render_policy(declared) is declared

    @pytest.mark.parametrize("value", ["eager", "deferred"])
    def test_operator_override_wins_over_the_declared_policy(self, monkeypatch, value):
        monkeypatch.setenv("MIMARSINAN_GUI_RESOURCE_RENDER", value)
        for declared in ResourceRenderPolicy:
            assert resolve_resource_render_policy(declared) is ResourceRenderPolicy(value)

    def test_unknown_override_fails_loud(self, monkeypatch):
        monkeypatch.setenv("MIMARSINAN_GUI_RESOURCE_RENDER", "sometimes")
        with pytest.raises(ValueError):
            resolve_resource_render_policy(ResourceRenderPolicy.EAGER)

    def test_every_descriptor_carries_a_persistable_source(self):
        descriptor = ResourceDescriptor(
            kind="ir_core_heatmap", rid="core/0",
            source=HeatmapSource(np.zeros((2, 2))), media_type="image/png",
        )
        assert descriptor.producer() == descriptor.source.render()
