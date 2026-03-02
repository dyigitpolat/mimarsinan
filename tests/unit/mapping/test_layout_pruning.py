"""Tests for LayoutIRMapping with pruning fraction applied."""

import pytest
import numpy as np
import torch

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from conftest import make_tiny_supermodel


class TestLayoutPruning:
    def test_zero_fraction_no_change(self):
        """With pruning_fraction=0, softcore dims should be identical to no-pruning baseline."""
        model = make_tiny_supermodel()
        
        baseline_mapper = LayoutIRMapping(max_axons=256, max_neurons=256)
        baseline_scs = baseline_mapper.collect_layout_softcores(model.get_mapper_repr())

        pruning_mapper = LayoutIRMapping(max_axons=256, max_neurons=256, pruning_fraction=0.0)
        pruning_scs = pruning_mapper.collect_layout_softcores(model.get_mapper_repr())

        assert len(baseline_scs) == len(pruning_scs)
        for b, p in zip(baseline_scs, pruning_scs):
            assert b.input_count == p.input_count
            assert b.output_count == p.output_count

    def test_pruning_reduces_softcore_dims(self):
        """With a non-zero pruning fraction, softcore dims should shrink."""
        model = make_tiny_supermodel()

        baseline_mapper = LayoutIRMapping(max_axons=256, max_neurons=256)
        baseline_scs = baseline_mapper.collect_layout_softcores(model.get_mapper_repr())

        pruning_mapper = LayoutIRMapping(max_axons=256, max_neurons=256, pruning_fraction=0.5)
        pruning_scs = pruning_mapper.collect_layout_softcores(model.get_mapper_repr())

        assert len(baseline_scs) == len(pruning_scs)
        # At least one softcore should be smaller
        any_smaller = False
        for b, p in zip(baseline_scs, pruning_scs):
            if p.input_count < b.input_count or p.output_count < b.output_count:
                any_smaller = True
            # Pruned should never be bigger
            assert p.input_count <= b.input_count
            assert p.output_count <= b.output_count
        assert any_smaller, "At least one softcore should shrink with pruning"

    def test_pruning_deterministic(self):
        """Same seed should produce identical pruned dims."""
        model = make_tiny_supermodel()

        mapper1 = LayoutIRMapping(max_axons=256, max_neurons=256, pruning_fraction=0.3, threshold_seed=42)
        scs1 = mapper1.collect_layout_softcores(model.get_mapper_repr())

        mapper2 = LayoutIRMapping(max_axons=256, max_neurons=256, pruning_fraction=0.3, threshold_seed=42)
        scs2 = mapper2.collect_layout_softcores(model.get_mapper_repr())

        assert len(scs1) == len(scs2)
        for s1, s2 in zip(scs1, scs2):
            assert s1.input_count == s2.input_count
            assert s1.output_count == s2.output_count

    def test_pruning_never_below_one(self):
        """Pruned dims should never drop below 1."""
        model = make_tiny_supermodel()

        mapper = LayoutIRMapping(max_axons=256, max_neurons=256, pruning_fraction=0.99)
        scs = mapper.collect_layout_softcores(model.get_mapper_repr())

        for sc in scs:
            assert sc.input_count >= 1, f"input_count should be >= 1, got {sc.input_count}"
            assert sc.output_count >= 1, f"output_count should be >= 1, got {sc.output_count}"

    def test_effective_fraction_is_80_percent(self):
        """The applied reduction should be 80% of the user-provided fraction."""
        model = make_tiny_supermodel()

        baseline_mapper = LayoutIRMapping(max_axons=256, max_neurons=256)
        baseline_scs = baseline_mapper.collect_layout_softcores(model.get_mapper_repr())

        # With fraction=1.0, effective is 0.8
        pruning_mapper = LayoutIRMapping(max_axons=256, max_neurons=256, pruning_fraction=1.0, threshold_seed=0)
        pruning_scs = pruning_mapper.collect_layout_softcores(model.get_mapper_repr())

        for b, p in zip(baseline_scs, pruning_scs):
            # Expected: input_count - floor(input_count * 0.8), but at least 1
            expected_in = max(1, b.input_count - int(b.input_count * 0.8))
            expected_out = max(1, b.output_count - int(b.output_count * 0.8))
            assert p.input_count == expected_in, \
                f"Expected input_count={expected_in}, got {p.input_count} (baseline={b.input_count})"
            assert p.output_count == expected_out, \
                f"Expected output_count={expected_out}, got {p.output_count} (baseline={b.output_count})"
