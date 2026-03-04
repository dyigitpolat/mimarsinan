"""Tests for centralized propagative pruning (compute_propagated_pruned_rows_cols)."""

import pytest
import numpy as np

from mimarsinan.mapping.pruning_propagation import compute_propagated_pruned_rows_cols


class TestComputePropagatedPrunedRowsCols:
    def test_initial_from_matrix_zero_threshold(self):
        """With no initial sets, pruned rows/cols are those below zero_threshold."""
        w = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        zero_rows, zero_cols = compute_propagated_pruned_rows_cols(w, zero_threshold=1e-8)
        assert zero_rows == {1}
        assert zero_cols == {1}

    def test_initial_sets_used_and_propagated(self):
        """When initial_zero_rows/cols are provided, propagation extends them."""
        # Row 0 only feeds col 0; col 0 only receives from row 0. So if we mark col 0 pruned,
        # row 0 should be pruned by propagation.
        w = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
            w,
            initial_zero_rows=set(),
            initial_zero_cols={0},
        )
        assert 0 in zero_rows, "Row 0 only feeds pruned col 0 -> should be pruned"
        assert zero_cols == {0}

    def test_propagative_fixpoint_expands_both(self):
        """A row that only feeds pruned cols is pruned; a col that only receives from pruned rows is pruned."""
        # Cols 2,3 are initially pruned (zero). Row 3 only has non-zero in cols 2,3 -> row 3 pruned.
        # Then col 1 only receives from row 1 (and row 1 is not pruned). So no extra col.
        # Row 3 only feeds 2,3 -> pruned.
        tiny = 1e-10
        w = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, tiny, tiny],
        ], dtype=np.float64)
        zero_rows, zero_cols = compute_propagated_pruned_rows_cols(w, zero_threshold=1e-8)
        assert 2 in zero_cols and 3 in zero_cols
        assert 3 in zero_rows

    def test_all_zero_returns_all_indices(self):
        """Fully zero matrix: all rows and cols pruned."""
        w = np.zeros((2, 3), dtype=np.float64)
        zero_rows, zero_cols = compute_propagated_pruned_rows_cols(w, zero_threshold=1e-8)
        assert zero_rows == {0, 1}
        assert zero_cols == {0, 1, 2}

    def test_none_pruned_when_all_above_threshold(self):
        """Matrix with all entries above threshold: no pruned rows/cols."""
        w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        zero_rows, zero_cols = compute_propagated_pruned_rows_cols(w, zero_threshold=1e-8)
        assert len(zero_rows) == 0
        assert len(zero_cols) == 0
