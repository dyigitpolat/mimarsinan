"""Tests for mapping_verifier: soft-core mapping verification and hardware config checking.

Verifies:
1. That layout mapping produces valid LayoutSoftCoreSpec results for native and
   torch-converted models.
2. That layout softcore count matches actual IRMapping neural core count (1-1 correspondence).
3. That layout softcore shapes match actual neural core matrix shapes.
4. That hardware config verification correctly identifies sufficient/insufficient configs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.mapping.mapping_verifier import (
    MappingVerificationResult,
    verify_soft_core_mapping,
    verify_hardware_config,
)
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import NeuralCore


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_native_model_repr():
    """Build a minimal native Supermodel and return its mapper repr."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from conftest import make_tiny_supermodel
    model = make_tiny_supermodel(input_shape=(1, 8, 8), num_classes=4)
    return model.get_mapper_repr()


def _make_torch_mlp_repr():
    """Build a simple torch MLP and return its mapper repr after conversion."""
    from mimarsinan.torch_mapping.converter import convert_torch_model

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 8)
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x.view(x.size(0), -1))))

    model = TinyMLP()
    supermodel = convert_torch_model(model, input_shape=(16,), num_classes=8)
    return supermodel.get_mapper_repr()


def _make_torch_conv_repr():
    """Build a simple conv net and return its mapper repr after conversion."""
    from mimarsinan.torch_mapping.converter import convert_torch_model

    class TinyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(4 * 4 * 4, 10)
        def forward(self, x):
            x = self.pool(self.relu(self.conv(x)))
            return self.fc(x.flatten(1))

    model = TinyConv()
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn(1, 1, 8, 8))
    supermodel = convert_torch_model(model, input_shape=(1, 8, 8), num_classes=10)
    return supermodel.get_mapper_repr()


def _make_torch_mlp_mixer_repr():
    """Build an MLP-Mixer and return its mapper repr after conversion."""
    from mimarsinan.models.builders import BUILDERS_REGISTRY
    from mimarsinan.torch_mapping.converter import convert_torch_model

    builder = BUILDERS_REGISTRY["mlp_mixer"](
        device=torch.device("cpu"),
        input_shape=(1, 28, 28),
        num_classes=10,
        pipeline_config={"target_tq": 32, "device": "cpu"},
    )
    model = builder.build({
        "patch_n_1": 4,
        "patch_m_1": 4,
        "patch_c_1": 32,
        "fc_w_1": 64,
        "fc_w_2": 64,
        "base_activation": "LeakyReLU",
    })
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn(1, 1, 28, 28))
    supermodel = convert_torch_model(
        model, input_shape=(1, 28, 28), num_classes=10, device="cpu", Tq=32
    )
    return supermodel.get_mapper_repr()


# ── Tests: verify_soft_core_mapping ────────────────────────────────────────

class TestVerifySoftCoreMapping:
    def test_native_model_feasible(self):
        repr_ = _make_native_model_repr()
        result = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256)
        assert isinstance(result, MappingVerificationResult)
        assert result.feasible
        assert result.error is None
        assert result.num_neural_cores > 0
        assert len(result.softcores) == result.num_neural_cores

    def test_native_model_softcore_shapes_positive(self):
        repr_ = _make_native_model_repr()
        result = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256)
        assert result.feasible
        for sc in result.softcores:
            assert sc.input_count > 0
            assert sc.output_count > 0

    def test_torch_mlp_feasible(self):
        repr_ = _make_torch_mlp_repr()
        result = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256)
        assert result.feasible
        assert result.num_neural_cores >= 1  # Identity layers become ComputeOps

    def test_torch_conv_feasible(self):
        repr_ = _make_torch_conv_repr()
        result = verify_soft_core_mapping(repr_, max_axons=512, max_neurons=512)
        assert result.feasible
        assert result.num_neural_cores >= 1  # Identity layers become ComputeOps

    def test_max_input_output_stats(self):
        repr_ = _make_native_model_repr()
        result = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256)
        assert result.max_input_size == max(sc.input_count for sc in result.softcores)
        assert result.max_output_size == max(sc.output_count for sc in result.softcores)
        assert result.total_area == sum(sc.area for sc in result.softcores)

    def test_pruning_reduces_softcores(self):
        repr_ = _make_native_model_repr()
        base = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256,
                                        pruning_fraction=0.0)
        pruned = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256,
                                          pruning_fraction=0.5)
        assert base.num_neural_cores == pruned.num_neural_cores
        # At least one dimension should shrink
        any_smaller = any(
            p.input_count < b.input_count or p.output_count < b.output_count
            for b, p in zip(base.softcores, pruned.softcores)
        )
        assert any_smaller

    def test_threshold_groups_assigned(self):
        repr_ = _make_native_model_repr()
        result = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256,
                                          threshold_groups=3)
        assert result.feasible
        groups = {sc.threshold_group_id for sc in result.softcores}
        # All groups should be valid indices (0 to threshold_groups-1)
        assert all(0 <= g <= 2 for g in groups)

    def test_deterministic_with_seed(self):
        repr_ = _make_native_model_repr()
        r1 = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256,
                                      threshold_groups=3, threshold_seed=42)
        r2 = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256,
                                      threshold_groups=3, threshold_seed=42)
        assert len(r1.softcores) == len(r2.softcores)
        for s1, s2 in zip(r1.softcores, r2.softcores):
            assert s1.threshold_group_id == s2.threshold_group_id

    def test_different_seeds_may_differ(self):
        repr_ = _make_native_model_repr()
        r1 = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256,
                                      threshold_groups=4, threshold_seed=0)
        r2 = verify_soft_core_mapping(repr_, max_axons=256, max_neurons=256,
                                      threshold_groups=4, threshold_seed=999)
        # With 4 groups and multiple cores, seeds should produce different assignments
        # (not guaranteed, but very likely)
        assert len(r1.softcores) == len(r2.softcores)

    def test_mlp_mixer_layout_preserves_multiple_neural_segments(self):
        """Layout pass should preserve neural depth across intervening ComputeOps."""
        repr_ = _make_torch_mlp_mixer_repr()
        result = verify_soft_core_mapping(repr_, max_axons=4096, max_neurons=4096)
        assert result.feasible
        latency_tags = {sc.latency_tag for sc in result.softcores if sc.latency_tag is not None}
        assert len(latency_tags) >= 4, (
            "MLP-Mixer should expose multiple sequential neural segments in layout "
            "verification; compute-op boundaries must not collapse all softcores "
            "to one latency tag."
        )

    def test_mlp_mixer_reports_host_side_segments(self):
        """MLP-Mixer layout should report compute-only segments separately."""
        repr_ = _make_torch_mlp_mixer_repr()
        result = verify_soft_core_mapping(repr_, max_axons=4096, max_neurons=4096)
        assert result.feasible
        assert result.host_side_segment_count == 5

    def test_mlp_mixer_exposes_layout_preview_flow(self):
        """Layout preview should alternate host runs and latency groups."""
        repr_ = _make_torch_mlp_mixer_repr()
        result = verify_soft_core_mapping(repr_, max_axons=4096, max_neurons=4096)
        assert result.feasible
        preview = result.layout_preview
        assert preview is not None
        flow = preview["flow"]
        assert flow[0]["kind"] == "input"
        assert flow[-1]["kind"] == "output"
        host_counts = [item["compute_op_count"] for item in flow if item["kind"] == "host"]
        neural_counts = [item["softcore_count"] for item in flow if item["kind"] == "neural"]
        latency_group_indices = [item["latency_group_index"] for item in flow if item["kind"] == "neural"]
        # Slot 0: patch_embed Conv2d (1 generic module ComputeOp)
        # Slots 1-4: mixer fc2 layers (per-column ComputeOps) + mean + classifier
        assert host_counts[0] == 1, (
            f"Patch embed Conv2d should create 1 module ComputeOp, got {host_counts[0]}"
        )
        assert neural_counts == [32, 16, 32, 16]
        assert latency_group_indices == [0, 1, 2, 3]


# ── Tests: Layout ↔ IR 1-1 Correspondence ──────────────────────────────────

class TestLayoutToIRCorrespondence:
    """Verify layout mapping matches actual IRMapping 1-1 in core count and shapes."""

    def _layout_and_ir_cores(self, repr_, max_axons=256, max_neurons=256):
        from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping

        layout = LayoutIRMapping(max_axons=max_axons, max_neurons=max_neurons)
        layout_scs = layout.collect_layout_softcores(repr_)

        ir = IRMapping(
            q_max=127.0,
            firing_mode="Default",
            max_axons=max_axons,
            max_neurons=max_neurons,
        )
        ir_graph = ir.map(repr_)
        ir_neural = ir_graph.get_neural_cores()

        return layout_scs, ir_neural

    def test_native_model_count_matches(self):
        repr_ = _make_native_model_repr()
        layout_scs, ir_neural = self._layout_and_ir_cores(repr_)
        assert len(layout_scs) == len(ir_neural), (
            f"Layout produced {len(layout_scs)} softcores but IR has {len(ir_neural)} neural cores"
        )

    def test_torch_mlp_count_matches(self):
        repr_ = _make_torch_mlp_repr()
        layout_scs, ir_neural = self._layout_and_ir_cores(repr_, max_axons=256, max_neurons=256)
        assert len(layout_scs) == len(ir_neural)

    def test_torch_conv_count_matches(self):
        repr_ = _make_torch_conv_repr()
        layout_scs, ir_neural = self._layout_and_ir_cores(repr_, max_axons=512, max_neurons=512)
        assert len(layout_scs) == len(ir_neural)

    def test_native_model_output_shapes_match(self):
        """Layout output_count should equal IR core output width."""
        repr_ = _make_native_model_repr()
        layout_scs, ir_neural = self._layout_and_ir_cores(repr_)

        for i, (lsc, nc) in enumerate(zip(layout_scs, ir_neural)):
            # IR core matrix is (axons, neurons); neurons = output_count
            ir_out = nc.core_matrix.shape[1]
            assert lsc.output_count == ir_out, (
                f"Core {i}: layout output_count={lsc.output_count} != IR neurons={ir_out}"
            )

    def test_native_model_input_shapes_match(self):
        """Layout input_count should equal IR core input width (axons)."""
        repr_ = _make_native_model_repr()
        layout_scs, ir_neural = self._layout_and_ir_cores(repr_)

        for i, (lsc, nc) in enumerate(zip(layout_scs, ir_neural)):
            ir_in = nc.core_matrix.shape[0]
            assert lsc.input_count == ir_in, (
                f"Core {i}: layout input_count={lsc.input_count} != IR axons={ir_in}"
            )


# ── Tests: verify_hardware_config ──────────────────────────────────────────

class TestVerifyHardwareConfig:
    @pytest.fixture
    def simple_softcores(self):
        return [
            LayoutSoftCoreSpec(input_count=16, output_count=8, threshold_group_id=0),
            LayoutSoftCoreSpec(input_count=9, output_count=4, threshold_group_id=0),
        ]

    def test_sufficient_config_passes(self, simple_softcores):
        core_types = [{"max_axons": 32, "max_neurons": 16, "count": 4}]
        result = verify_hardware_config(simple_softcores, core_types)
        assert result["feasible"]
        assert result["errors"] == []

    def test_insufficient_axons_fails(self, simple_softcores):
        # max_axons=8 < required 16; no type fits the largest softcore
        core_types = [{"max_axons": 8, "max_neurons": 16, "count": 10}]
        result = verify_hardware_config(simple_softcores, core_types)
        assert not result["feasible"]
        assert any("axons" in e.lower() for e in result["errors"])
        # With multi-type check we use "core_types" when no type covers the largest
        assert len(result["field_errors"]) > 0

    def test_insufficient_neurons_fails(self, simple_softcores):
        # max_neurons=4 < required 8; no type fits the largest softcore
        core_types = [{"max_axons": 32, "max_neurons": 4, "count": 10}]
        result = verify_hardware_config(simple_softcores, core_types)
        assert not result["feasible"]
        assert any("neurons" in e.lower() for e in result["errors"])
        assert len(result["field_errors"]) > 0

    def test_too_few_cores_fails(self):
        # 20 softcores, only 1 core available
        softcores = [
            LayoutSoftCoreSpec(input_count=16, output_count=8, threshold_group_id=0)
            for _ in range(20)
        ]
        core_types = [{"max_axons": 32, "max_neurons": 16, "count": 1}]
        result = verify_hardware_config(softcores, core_types)
        assert not result["feasible"]

    def test_no_softcores_fails(self):
        result = verify_hardware_config([], [{"max_axons": 32, "max_neurons": 16, "count": 4}])
        assert not result["feasible"]

    def test_no_core_types_fails(self, simple_softcores):
        result = verify_hardware_config(simple_softcores, [])
        assert not result["feasible"]
        assert "core_types" in result["field_errors"]

    def test_exact_fit_passes(self):
        # One softcore, one core with exact dimensions
        softcores = [LayoutSoftCoreSpec(input_count=16, output_count=8, threshold_group_id=0)]
        core_types = [{"max_axons": 16, "max_neurons": 8, "count": 1}]
        result = verify_hardware_config(softcores, core_types)
        assert result["feasible"]

    def test_packing_result_returned(self, simple_softcores):
        core_types = [{"max_axons": 32, "max_neurons": 16, "count": 4}]
        result = verify_hardware_config(simple_softcores, core_types)
        assert result["packing_result"] is not None
        assert result["packing_result"].cores_used >= 1

    def test_many_softcores_fewer_cores_passes(self):
        """Regression: many softcores CAN fit into fewer hardware cores via bin-packing.
        The old code incorrectly required total_count >= len(softcores). Multiple softcores
        can share a single hardware core, so the packer is the sole arbiter of feasibility.
        """
        # 10 softcores, each (4, 4). A core of (16, 16) can hold many of them.
        # 3 cores of (16, 16) should pack all 10 softcores easily.
        softcores = [
            LayoutSoftCoreSpec(input_count=4, output_count=4, threshold_group_id=0)
            for _ in range(10)
        ]
        core_types = [{"max_axons": 16, "max_neurons": 16, "count": 3}]
        result = verify_hardware_config(softcores, core_types)
        assert result["feasible"], (
            f"Expected feasible but got errors: {result['errors']}"
        )

    def test_total_count_less_than_softcores_can_still_pass(self):
        """Regression: total core count < len(softcores) must NOT be a pre-check error.
        The packer decides, not a headcount comparison."""
        # 20 softcores that fit 4-per-core → need only 5 cores
        softcores = [
            LayoutSoftCoreSpec(input_count=3, output_count=3, threshold_group_id=0)
            for _ in range(20)
        ]
        core_types = [{"max_axons": 16, "max_neurons": 16, "count": 6}]  # 6 < 20
        result = verify_hardware_config(softcores, core_types)
        assert result["feasible"], (
            f"6 cores should pack 20 small softcores but got: {result['errors']}"
        )

    def test_count_field_error_only_on_packing_failure(self):
        """total_count field_error should only appear when packing genuinely fails."""
        softcores = [
            LayoutSoftCoreSpec(input_count=4, output_count=4, threshold_group_id=0)
            for _ in range(10)
        ]
        # 1 core: too few to pack 10 softcores
        result_fail = verify_hardware_config(
            softcores, [{"max_axons": 16, "max_neurons": 16, "count": 1}]
        )
        assert not result_fail["feasible"]
        assert "total_count" in result_fail["field_errors"]

        # 5 cores: plenty to pack (10 softcores * 4*4 area, core capacity 16*16=256, fits 16 per core)
        result_pass = verify_hardware_config(
            softcores, [{"max_axons": 16, "max_neurons": 16, "count": 5}]
        )
        assert result_pass["feasible"]
        assert "total_count" not in result_pass["field_errors"]

    def test_two_type_config_at_least_one_covers_largest_passes(self):
        """With two core types (e.g. H×W and W×H), only one type need fit the largest softcore."""
        # Largest softcore is (20, 10). Type (10, 20) does not fit it; type (20, 10) does.
        softcores = [
            LayoutSoftCoreSpec(input_count=20, output_count=10, threshold_group_id=0),
            LayoutSoftCoreSpec(input_count=8, output_count=8, threshold_group_id=0),
        ]
        core_types = [
            {"max_axons": 10, "max_neurons": 20, "count": 2},
            {"max_axons": 20, "max_neurons": 10, "count": 2},
        ]
        result = verify_hardware_config(softcores, core_types)
        assert result["feasible"], (
            f"Two-type config with one type covering (20,10) should pass: {result['errors']}"
        )
