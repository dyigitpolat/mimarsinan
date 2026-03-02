"""Tests for pruning mask computation and weight pruning application."""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.layers import LeakyGradReLU
from mimarsinan.transformations.pruning import (
    compute_pruning_masks,
    apply_pruning_masks,
)


class TestComputePruningMasks:
    def _make_perceptron(self, out_features=8, in_features=16):
        p = Perceptron(out_features, in_features)
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        return p

    def test_returns_correct_shapes(self):
        p = self._make_perceptron(8, 16)
        row_mask, col_mask = compute_pruning_masks(p, pruning_fraction=0.25)
        assert row_mask.shape == (8,), f"Expected (8,), got {row_mask.shape}"
        assert col_mask.shape == (16,), f"Expected (16,), got {col_mask.shape}"

    def test_fraction_zero_keeps_all(self):
        p = self._make_perceptron(8, 16)
        row_mask, col_mask = compute_pruning_masks(p, pruning_fraction=0.0)
        assert row_mask.all(), "All rows should be kept with fraction=0"
        assert col_mask.all(), "All cols should be kept with fraction=0"

    def test_fraction_one_removes_all(self):
        p = self._make_perceptron(8, 16)
        row_mask, col_mask = compute_pruning_masks(p, pruning_fraction=1.0)
        assert not row_mask.any(), "All rows should be pruned with fraction=1"
        assert not col_mask.any(), "All cols should be pruned with fraction=1"

    def test_identifies_smallest_rows(self):
        """With known weights, verify correct rows are identified as least significant."""
        p = self._make_perceptron(4, 4)
        # Set weights so rows 0,1 are small and rows 2,3 are large
        with torch.no_grad():
            p.layer.weight.data = torch.tensor([
                [0.01, 0.01, 0.01, 0.01],  # row 0: small
                [0.02, 0.02, 0.02, 0.02],  # row 1: small
                [1.00, 1.00, 1.00, 1.00],  # row 2: large
                [2.00, 2.00, 2.00, 2.00],  # row 3: large
            ])
        row_mask, _ = compute_pruning_masks(p, pruning_fraction=0.5)
        # Rows 0 and 1 should be pruned (False), rows 2 and 3 kept (True)
        assert not row_mask[0].item(), "Row 0 (smallest) should be pruned"
        assert not row_mask[1].item(), "Row 1 (2nd smallest) should be pruned"
        assert row_mask[2].item(), "Row 2 (large) should be kept"
        assert row_mask[3].item(), "Row 3 (largest) should be kept"

    def test_identifies_smallest_columns(self):
        """With known weights, verify correct columns are identified as least significant."""
        p = self._make_perceptron(4, 4)
        with torch.no_grad():
            p.layer.weight.data = torch.tensor([
                [0.01, 1.00, 0.02, 2.00],
                [0.01, 1.00, 0.02, 2.00],
                [0.01, 1.00, 0.02, 2.00],
                [0.01, 1.00, 0.02, 2.00],
            ])
        _, col_mask = compute_pruning_masks(p, pruning_fraction=0.5)
        # Cols 0 and 2 should be pruned (False), cols 1 and 3 kept (True)
        assert not col_mask[0].item(), "Col 0 (smallest) should be pruned"
        assert col_mask[1].item(), "Col 1 (large) should be kept"
        assert not col_mask[2].item(), "Col 2 (2nd smallest) should be pruned"
        assert col_mask[3].item(), "Col 3 (largest) should be kept"

    def test_masks_are_boolean(self):
        p = self._make_perceptron(8, 16)
        row_mask, col_mask = compute_pruning_masks(p, pruning_fraction=0.5)
        assert row_mask.dtype == torch.bool
        assert col_mask.dtype == torch.bool

    def test_correct_count_pruned(self):
        """25% of 8 rows = 2 rows pruned, 25% of 16 cols = 4 cols pruned."""
        p = self._make_perceptron(8, 16)
        row_mask, col_mask = compute_pruning_masks(p, pruning_fraction=0.25)
        assert row_mask.sum().item() == 6  # 8 - 2 kept
        assert col_mask.sum().item() == 12  # 16 - 4 kept


class TestApplyPruningMasks:
    def _make_perceptron(self, out_features=4, in_features=4):
        p = Perceptron(out_features, in_features)
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        return p

    def test_rate_zero_no_change(self):
        p = self._make_perceptron()
        original_weights = p.layer.weight.data.clone()
        row_mask = torch.tensor([True, False, True, False])
        col_mask = torch.tensor([True, False, True, False])
        apply_pruning_masks(p, row_mask, col_mask, rate=0.0)
        assert torch.allclose(p.layer.weight.data, original_weights), \
            "Rate=0 should not change weights"

    def test_rate_one_zeros_pruned(self):
        p = self._make_perceptron()
        with torch.no_grad():
            p.layer.weight.data.fill_(1.0)
        row_mask = torch.tensor([True, False, True, True])  # row 1 pruned
        col_mask = torch.tensor([True, True, True, True])   # no cols pruned
        apply_pruning_masks(p, row_mask, col_mask, rate=1.0)
        # Row 1 should be zeroed
        assert (p.layer.weight.data[1] == 0.0).all(), "Pruned row should be zeroed at rate=1"
        # Other rows should be unchanged
        assert (p.layer.weight.data[0] == 1.0).all(), "Kept row 0 should be unchanged"
        assert (p.layer.weight.data[2] == 1.0).all(), "Kept row 2 should be unchanged"
        assert (p.layer.weight.data[3] == 1.0).all(), "Kept row 3 should be unchanged"

    def test_rate_one_zeros_pruned_columns(self):
        p = self._make_perceptron()
        with torch.no_grad():
            p.layer.weight.data.fill_(1.0)
        row_mask = torch.tensor([True, True, True, True])   # no rows pruned
        col_mask = torch.tensor([True, False, True, True])  # col 1 pruned
        apply_pruning_masks(p, row_mask, col_mask, rate=1.0)
        # Col 1 should be zeroed
        assert (p.layer.weight.data[:, 1] == 0.0).all(), "Pruned col should be zeroed at rate=1"
        # Other cols should be unchanged
        assert (p.layer.weight.data[:, 0] == 1.0).all()
        assert (p.layer.weight.data[:, 2] == 1.0).all()
        assert (p.layer.weight.data[:, 3] == 1.0).all()

    def test_partial_rate(self):
        p = self._make_perceptron()
        with torch.no_grad():
            p.layer.weight.data.fill_(1.0)
        row_mask = torch.tensor([True, False, True, True])  # row 1 pruned
        col_mask = torch.tensor([True, True, True, True])   # no cols pruned
        apply_pruning_masks(p, row_mask, col_mask, rate=0.3)
        # Row 1 should be multiplied by (1 - 0.3) = 0.7
        expected = 0.7
        assert torch.allclose(p.layer.weight.data[1], torch.full((4,), expected)), \
            f"Pruned row at rate=0.3 should be {expected}"
        # Other rows should be unchanged
        assert (p.layer.weight.data[0] == 1.0).all()

    def test_both_row_and_col_pruned(self):
        """When both a row and column are pruned, the intersection weight should still scale by (1-rate)."""
        p = self._make_perceptron()
        with torch.no_grad():
            p.layer.weight.data.fill_(1.0)
        row_mask = torch.tensor([True, False, True, True])  # row 1 pruned
        col_mask = torch.tensor([True, False, True, True])  # col 1 pruned
        apply_pruning_masks(p, row_mask, col_mask, rate=1.0)
        # Weight at [1,1] should be 0 (both pruned)
        assert p.layer.weight.data[1, 1].item() == 0.0
        # Weight at [0,1] should be 0 (col pruned)
        assert p.layer.weight.data[0, 1].item() == 0.0
        # Weight at [1,0] should be 0 (row pruned)
        assert p.layer.weight.data[1, 0].item() == 0.0
        # Weight at [0,0] should be 1 (neither pruned)
        assert p.layer.weight.data[0, 0].item() == 1.0
