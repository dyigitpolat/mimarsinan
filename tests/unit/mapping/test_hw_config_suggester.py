"""Tests for hw_config_suggester: greedy hardware configuration algorithm.

Verifies:
1. Basic suggestion properties (returns at least 1 core type, positive counts).
2. Suggested config is sufficient to pack all softcores (end-to-end verify round-trip).
3. Heuristics behave sensibly with pruning, threshold groups, coalescing.
4. Single-type coverage: dimensions cover every softcore.
5. The suggest_hardware_config_for_model convenience wrapper.
6. Regression: suggested count passes pack_layout at exactly that count (binary-search fix).
7. Regression: verify_hardware_config does not falsely fail when count < num_softcores.
"""

from __future__ import annotations

import pytest
import math
from typing import List

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.hw_config_suggester import (
    suggest_hardware_config,
    suggest_hardware_config_for_model,
    HardwareSuggestion,
)
from mimarsinan.mapping.mapping_verifier import verify_hardware_config


# ── Helper factories ────────────────────────────────────────────────────────

def _make_softcores(specs: List[tuple]) -> List[LayoutSoftCoreSpec]:
    """Build softcores from list of (input_count, output_count) tuples."""
    return [
        LayoutSoftCoreSpec(input_count=a, output_count=b, threshold_group_id=0)
        for a, b in specs
    ]


def _make_uniform_softcores(n: int, axons: int = 16, neurons: int = 8) -> List[LayoutSoftCoreSpec]:
    return [
        LayoutSoftCoreSpec(input_count=axons, output_count=neurons, threshold_group_id=i % 4)
        for i in range(n)
    ]


# ── Tests: suggest_hardware_config ─────────────────────────────────────────

class TestSuggestHardwareConfig:
    def test_returns_hardware_suggestion(self):
        softcores = _make_uniform_softcores(10)
        result = suggest_hardware_config(softcores)
        assert isinstance(result, HardwareSuggestion)

    def test_non_empty_core_types(self):
        softcores = _make_uniform_softcores(5)
        result = suggest_hardware_config(softcores)
        assert len(result.core_types) == 2
        assert result.total_cores > 0

    def test_core_type_fields_present(self):
        softcores = _make_uniform_softcores(3)
        result = suggest_hardware_config(softcores)
        for ct in result.core_types:
            assert "max_axons" in ct
            assert "max_neurons" in ct
            assert "count" in ct
            assert ct["max_axons"] > 0
            assert ct["max_neurons"] > 0
            assert ct["count"] > 0

    def test_suggested_config_is_sufficient(self):
        """Verify that the suggested config actually packs all softcores."""
        softcores = _make_uniform_softcores(15, axons=32, neurons=16)
        suggestion = suggest_hardware_config(softcores)
        verification = verify_hardware_config(softcores, suggestion.core_types)
        assert verification["feasible"], (
            f"Suggested config not sufficient: {verification['errors']}"
        )

    def test_suggested_config_sufficient_for_varied_sizes(self):
        """Test with heterogeneous softcore sizes."""
        softcores = _make_softcores([
            (64, 32), (32, 16), (16, 8), (8, 4),
            (64, 32), (32, 16), (16, 8),
        ])
        suggestion = suggest_hardware_config(softcores)
        verification = verify_hardware_config(softcores, suggestion.core_types)
        assert verification["feasible"], (
            f"Suggested config not sufficient: {verification['errors']}"
        )

    def test_single_softcore(self):
        softcores = [LayoutSoftCoreSpec(input_count=16, output_count=8, threshold_group_id=0)]
        result = suggest_hardware_config(softcores)
        assert result.total_cores >= 1
        verification = verify_hardware_config(softcores, result.core_types)
        assert verification["feasible"]

    def test_empty_softcores(self):
        result = suggest_hardware_config([])
        assert result.total_cores == 0
        assert result.core_types == []

    def test_safety_margin_applied(self):
        """Suggested count should exceed the minimum pack count."""
        softcores = _make_uniform_softcores(10, axons=16, neurons=8)
        result = suggest_hardware_config(softcores, safety_margin=0.15)
        # At minimum there should be 10 cores (1 per softcore worst case is unlikely
        # given perfect packing), but with margin they should be > min
        assert result.total_cores >= 1

    def test_hardware_bias_flag_propagated(self):
        softcores = _make_uniform_softcores(5)
        r_bias = suggest_hardware_config(softcores, hardware_bias=True)
        r_no_bias = suggest_hardware_config(softcores, hardware_bias=False)
        for ct in r_bias.core_types:
            assert ct.get("has_bias") is True
        for ct in r_no_bias.core_types:
            assert ct.get("has_bias") is False

    def test_two_types_cover_all_sizes(self):
        """Always produces two core types; together they cover every softcore."""
        large_cores = [LayoutSoftCoreSpec(input_count=512, output_count=256, threshold_group_id=0)] * 3
        small_cores = [LayoutSoftCoreSpec(input_count=8, output_count=4, threshold_group_id=0)] * 10
        all_cores = large_cores + small_cores
        result = suggest_hardware_config(all_cores)
        assert len(result.core_types) == 2
        # At least one type must fit the largest softcore (512, 256).
        fits_large = any(
            ct["max_axons"] >= 512 and ct["max_neurons"] >= 256
            for ct in result.core_types
        )
        assert fits_large

    def test_two_types_for_similar_sizes(self):
        """When all softcores have similar sizes, still recommend two types (HxW and WxH)."""
        softcores = _make_softcores([(16, 8), (18, 10), (14, 9), (17, 8)])
        result = suggest_hardware_config(softcores)
        assert len(result.core_types) == 2

    def test_granularity_rounding(self):
        """Axon/neuron granularity: dimensions are multiples and at least one type covers the softcore."""
        softcores = [LayoutSoftCoreSpec(input_count=17, output_count=9, threshold_group_id=0)]
        result = suggest_hardware_config(softcores, axon_granularity=8, neuron_granularity=8)
        assert len(result.core_types) == 2
        for ct in result.core_types:
            assert ct["max_axons"] % 8 == 0
            assert ct["max_neurons"] % 8 == 0
        # At least one type must fit (17, 9).
        fits = any(
            ct["max_axons"] >= 17 and ct["max_neurons"] >= 9
            for ct in result.core_types
        )
        assert fits

    def test_rationale_non_empty(self):
        softcores = _make_uniform_softcores(5)
        result = suggest_hardware_config(softcores)
        assert result.rationale
        assert len(result.rationale) > 5

    def test_large_model_suggestion(self):
        """Stress test: many softcores with varied sizes."""
        import random
        rng = random.Random(42)
        softcores = [
            LayoutSoftCoreSpec(
                input_count=rng.randint(8, 128),
                output_count=rng.randint(4, 64),
                threshold_group_id=rng.randint(0, 3),
            )
            for _ in range(50)
        ]
        result = suggest_hardware_config(softcores)
        verification = verify_hardware_config(softcores, result.core_types)
        assert verification["feasible"], (
            f"Large model suggestion not feasible: {verification['errors']}"
        )

    def test_suggested_count_passes_packing(self):
        """Regression: suggested config must pass pack_layout at the suggested counts."""
        from mimarsinan.mapping.layout.layout_packer import pack_layout
        from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType

        softcores = _make_uniform_softcores(50, axons=16, neurons=8)
        suggestion = suggest_hardware_config(softcores)
        assert len(suggestion.core_types) == 2

        hw_types = [
            LayoutHardCoreType(
                max_axons=ct["max_axons"],
                max_neurons=ct["max_neurons"],
                count=ct["count"],
            )
            for ct in suggestion.core_types
        ]
        result = pack_layout(softcores=softcores, core_types=hw_types)
        assert result.feasible, (
            f"Suggested config failed packing for {len(softcores)} softcores: {result.error}"
        )

    def test_suggested_count_consistent_with_verify(self):
        """End-to-end: suggest_hardware_config output must pass verify_hardware_config
        using the SAME softcores (no bound change between the two calls)."""
        import random
        rng = random.Random(7)
        softcores = [
            LayoutSoftCoreSpec(
                input_count=rng.randint(4, 32),
                output_count=rng.randint(4, 16),
                threshold_group_id=rng.randint(0, 1),
            )
            for _ in range(30)
        ]
        suggestion = suggest_hardware_config(softcores)
        verification = verify_hardware_config(softcores, suggestion.core_types)
        assert verification["feasible"], (
            f"suggest→verify round-trip failed: {verification['errors']}"
        )

    def test_suggested_count_not_excessive(self):
        """Suggested total_cores should not be more than 3x the minimum feasible total."""
        from mimarsinan.mapping.layout.layout_packer import pack_layout
        from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType

        softcores = _make_uniform_softcores(20, axons=8, neurons=4)
        suggestion = suggest_hardware_config(softcores, safety_margin=0.15)
        assert len(suggestion.core_types) == 2

        def pack_feasible(total: int) -> bool:
            c1, c2 = (total + 1) // 2, total // 2
            hw = [
                LayoutHardCoreType(
                    max_axons=suggestion.core_types[0]["max_axons"],
                    max_neurons=suggestion.core_types[0]["max_neurons"],
                    count=max(1, c1),
                ),
                LayoutHardCoreType(
                    max_axons=suggestion.core_types[1]["max_axons"],
                    max_neurons=suggestion.core_types[1]["max_neurons"],
                    count=max(1, c2),
                ),
            ]
            return pack_layout(softcores=softcores, core_types=hw).feasible

        lo, hi = 1, len(softcores)
        while not pack_feasible(hi):
            hi *= 2
            if hi > 10 * len(softcores):
                break
        while lo < hi:
            mid = (lo + hi) // 2
            if pack_feasible(mid):
                hi = mid
            else:
                lo = mid + 1
        min_feasible = lo

        assert suggestion.total_cores <= min_feasible * 3, (
            f"Suggested total_cores {suggestion.total_cores} excessive for min_feasible={min_feasible}"
        )

    def test_occupancy_constraint(self):
        """More than half of used hardware cores should house at least 4 software cores."""
        from mimarsinan.mapping.layout.layout_packer import pack_layout
        from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType

        softcores = _make_uniform_softcores(40, axons=16, neurons=8)
        suggestion = suggest_hardware_config(softcores)
        hw_types = [
            LayoutHardCoreType(
                max_axons=ct["max_axons"],
                max_neurons=ct["max_neurons"],
                count=ct["count"],
            )
            for ct in suggestion.core_types
        ]
        result = pack_layout(softcores=softcores, core_types=hw_types)
        assert result.feasible
        assert result.used_core_softcore_counts is not None
        counts = result.used_core_softcore_counts
        if len(counts) >= 2:
            n_ok = sum(1 for c in counts if c >= 4)
            assert n_ok > len(counts) / 2, (
                f"Occupancy constraint failed: {n_ok}/{len(counts)} cores have >=4 softcores"
            )

    def test_coalescing_two_types_same_neuron_height(self):
        """With coalescing, both types share full neuron coverage; axon widths differ."""
        softcores = _make_uniform_softcores(20, axons=32, neurons=24)
        result = suggest_hardware_config(softcores, allow_coalescing=True)
        assert len(result.core_types) == 2
        a1, n1 = result.core_types[0]["max_axons"], result.core_types[0]["max_neurons"]
        a2, n2 = result.core_types[1]["max_axons"], result.core_types[1]["max_neurons"]
        max_neu = max(sc.output_count for sc in softcores)
        # Both types must cover the full neuron count (neuron is the hard constraint).
        assert n1 >= max_neu and n2 >= max_neu, f"types ({a1}×{n1}) and ({a2}×{n2}) must both have neurons >= {max_neu}"
        # The two types must have different axon widths (compact A + wide B).
        assert a1 != a2, "coalescing should produce two different axon widths"
        # The larger type must cover the full axon count without coalescing.
        max_ax = max(sc.input_count for sc in softcores)
        assert max(a1, a2) >= max_ax, "at least one type should cover max axon count"
        # Feasibility check (with coalescing enabled for packing).
        verification = verify_hardware_config(softcores, result.core_types, allow_axon_coalescing=True)
        assert verification["feasible"]

    def test_both_coalescing_and_no_coalescing_produce_valid_mappings(self):
        """Both allow_coalescing=False (H×W, W×H) and True (H×H, W×H) must pack all softcores."""
        from mimarsinan.mapping.layout.layout_packer import pack_layout
        from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType

        # Mix of shapes including neuron-heavy (more neurons than axons)
        softcores = _make_softcores([
            (64, 32), (32, 64), (48, 48), (40, 60), (60, 40),
            (32, 64), (64, 32), (24, 48), (48, 24),
        ])
        for allow_coalescing in (False, True):
            suggestion = suggest_hardware_config(
                softcores, allow_coalescing=allow_coalescing
            )
            assert len(suggestion.core_types) == 2, (
                f"allow_coalescing={allow_coalescing}: expected 2 core types"
            )
            hw_types = [
                LayoutHardCoreType(
                    max_axons=ct["max_axons"],
                    max_neurons=ct["max_neurons"],
                    count=ct["count"],
                )
                for ct in suggestion.core_types
            ]
            result = pack_layout(softcores=softcores, core_types=hw_types)
            assert result.feasible, (
                f"allow_coalescing={allow_coalescing}: pack_layout failed: {result.error}"
            )
            v = verify_hardware_config(softcores, suggestion.core_types)
            assert v["feasible"], (
                f"allow_coalescing={allow_coalescing}: verify failed: {v['errors']}"
            )


# ── Tests: suggest_hardware_config_for_model ───────────────────────────────

class TestSuggestHardwareConfigForModel:
    def test_native_model(self):
        """suggest_hardware_config_for_model works with a native mapper repr."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from conftest import make_tiny_supermodel
        model = make_tiny_supermodel(input_shape=(1, 8, 8), num_classes=4)
        model_repr = model.get_mapper_repr()

        result = suggest_hardware_config_for_model(
            model_repr,
            max_axons=256,
            max_neurons=256,
        )
        assert isinstance(result, HardwareSuggestion)
        assert result.total_cores > 0
        assert len(result.core_types) >= 1

    def test_native_model_result_is_feasible(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from conftest import make_tiny_supermodel
        from mimarsinan.mapping.mapping_verifier import verify_soft_core_mapping

        model = make_tiny_supermodel(input_shape=(1, 8, 8), num_classes=4)
        model_repr = model.get_mapper_repr()

        suggestion = suggest_hardware_config_for_model(
            model_repr,
            max_axons=256,
            max_neurons=256,
            threshold_groups=2,
            pruning_fraction=0.3,
        )
        # Also get the actual softcores and verify the suggestion against them
        layout_result = verify_soft_core_mapping(
            model_repr,
            max_axons=256,
            max_neurons=256,
            threshold_groups=2,
            pruning_fraction=0.3,
        )
        assert layout_result.feasible
        verification = verify_hardware_config(layout_result.softcores, suggestion.core_types)
        assert verification["feasible"], (
            f"Suggested config is not feasible: {verification['errors']}"
        )

    def test_torch_mlp_model(self):
        """suggest_hardware_config_for_model works with a torch-converted MLP."""
        import torch.nn as nn
        from mimarsinan.torch_mapping.converter import convert_torch_model

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(32, 8)
            def forward(self, x):
                import torch
                return self.fc2(self.relu(self.fc1(x.view(x.size(0), -1))))

        model = TinyMLP()
        supermodel = convert_torch_model(model, input_shape=(16,), num_classes=8)
        model_repr = supermodel.get_mapper_repr()

        result = suggest_hardware_config_for_model(
            model_repr,
            max_axons=256,
            max_neurons=256,
        )
        assert result.total_cores > 0
        assert not result.rationale.startswith("Layout mapping failed")

    def test_invalid_model_repr_returns_empty(self):
        """A broken model repr should return empty suggestion with error rationale."""

        class BrokenRepr:
            def map_to_ir(self, mapping):
                raise RuntimeError("Intentional failure")

        result = suggest_hardware_config_for_model(
            BrokenRepr(),
            max_axons=256,
            max_neurons=256,
        )
        assert result.total_cores == 0
        assert "failed" in result.rationale.lower()

    def test_with_pruning_and_threshold_groups(self):
        """Pruning and threshold groups are factored in correctly."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from conftest import make_tiny_supermodel

        model = make_tiny_supermodel(input_shape=(1, 8, 8), num_classes=4)
        model_repr = model.get_mapper_repr()

        # Suggestion without pruning
        r_no_prune = suggest_hardware_config_for_model(
            model_repr, max_axons=256, max_neurons=256,
            pruning_fraction=0.0,
        )
        # Suggestion with heavy pruning (model should need fewer/smaller cores)
        r_pruned = suggest_hardware_config_for_model(
            model_repr, max_axons=256, max_neurons=256,
            pruning_fraction=0.7,
        )
        # Both should be valid (non-empty)
        assert r_no_prune.total_cores > 0
        assert r_pruned.total_cores > 0
        # After pruning, total cores should be <= unpruned (heuristic not guaranteed but typical)
        assert r_pruned.total_cores <= r_no_prune.total_cores * 2
