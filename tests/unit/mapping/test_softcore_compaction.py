"""Tests for ``compact_soft_core_mapping``'s post-liveness contract.

The dead-row + dead-col branch is unreachable: ``compute_liveness +
IRGraph.remove_nodes`` deletes such cores from the IR graph before
soft-core mapping runs. The compactor asserts that contract and raises
when violated -- this test pins the assertion message so a regression
shows up loudly.

The legitimate BIAS_ONLY branch (every axon dead, some neuron columns
alive via ``hardware_bias``) is preserved end-to-end: collapsing to a
single OFF-source axon while keeping every live neuron column.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore_mapping import (
    SoftCore,
    compact_soft_core_mapping,
)


def _bias_only_softcore(*, n_axons: int, n_cols: int) -> SoftCore:
    """Build a SoftCore whose every axon is dead but bias columns survive."""
    sc = SoftCore(
        core_matrix=np.zeros((n_axons, n_cols), dtype=np.float64),
        axon_sources=[
            SpikeSource(-1, 0, is_input=False, is_off=True)
            for _ in range(n_axons)
        ],
        id=0,
    )
    sc.hardware_bias = np.array([2.0] * n_cols, dtype=np.float64)
    sc.pruned_row_mask = [True] * n_axons
    sc.pruned_col_mask = [False] * n_cols
    return sc


def _dead_softcore(*, n_axons: int, n_cols: int) -> SoftCore:
    """Build a SoftCore whose every axon AND every column is pruned.

    With a working liveness pass this case never reaches the compactor;
    we use it only to assert the regression-detection ``AssertionError``.
    """
    sc = SoftCore(
        core_matrix=np.zeros((n_axons, n_cols), dtype=np.float64),
        axon_sources=[
            SpikeSource(-1, 0, is_input=False, is_off=True)
            for _ in range(n_axons)
        ],
        id=42,
    )
    sc.pruned_row_mask = [True] * n_axons
    sc.pruned_col_mask = [True] * n_cols
    return sc


class TestBiasOnlyCompaction:
    def test_compact_preserves_bias_only_columns(self):
        """A SoftCore with all rows pruned but live bias-only columns
        collapses to ``(1, n_cols)`` with a single OFF-source axon. Every
        bias entry survives and downstream consumers can still address each
        live neuron via the returned ``reindex_maps``.
        """
        sc = _bias_only_softcore(n_axons=4, n_cols=3)
        # A second consumer core that reads sc.neuron 0 and sc.neuron 2.
        consumer = SoftCore(
            core_matrix=np.array([[1.0], [1.0]], dtype=np.float64),
            axon_sources=[
                SpikeSource(0, 0, is_input=False, is_off=False),
                SpikeSource(0, 2, is_input=False, is_off=False),
            ],
            id=1,
        )
        consumer.pruned_row_mask = [False, False]
        consumer.pruned_col_mask = [False]
        outputs = [SpikeSource(1, 0, is_input=False, is_off=False)]

        reindex = compact_soft_core_mapping([sc, consumer], outputs)

        # bias-only collapse: 1 OFF axon row, all live neuron columns kept
        assert sc.core_matrix.shape == (1, 3)
        assert len(sc.axon_sources) == 1
        assert sc.axon_sources[0].is_off_
        assert sc.hardware_bias.tolist() == [2.0, 2.0, 2.0]

        # consumer reindex map: neuron 0 -> 0, neuron 2 -> 2 (no compaction)
        assert reindex[0] == {0: 0, 1: 1, 2: 2}
        # consumer's references to sc.neuron 0 and sc.neuron 2 stay valid
        assert consumer.axon_sources[0].neuron_ == 0
        assert consumer.axon_sources[1].neuron_ == 2


class TestDeadPathUnreachable:
    def test_compact_drops_dead_path_unreachable(self):
        """A SoftCore whose every column is pruned must never reach the
        compactor. ``compute_liveness + remove_nodes`` is responsible for
        deleting it from the IR graph; reaching this branch indicates a
        regression and the compactor raises ``AssertionError`` with a
        message pointing at the liveness pass.
        """
        sc = _dead_softcore(n_axons=2, n_cols=2)
        with pytest.raises(AssertionError, match=r"[Ll]iveness"):
            compact_soft_core_mapping([sc], [])
