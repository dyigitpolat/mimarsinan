"""The cross-core wave operators as array reductions.

Each kernel is the one-for-one twin of a reference kernel and computes the
IDENTICAL predicate over identical inputs -- pure set membership, no floating
point anywhere, so bit-identity is by construction and the differential tests
in ``test_flat_kernels.py`` hold both sides equal on every topology.
"""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.pruning.graph.flat.state import FlatState

__all__ = ["flat_cross_core_dead_axons", "flat_orphan_neurons"]


def flat_cross_core_dead_axons(state: FlatState, col_dead: np.ndarray) -> np.ndarray:
    """Twin of ``_cross_core_dead_axons`` over the whole graph at once.

    OFF axons are always dead; a direct axon dies with its producer port; a
    relayed axon dies when EVERY transfer-mapped producer port is dead.
    Returns a boolean over flat axon rows.
    """
    dead = np.zeros(state.n_rows, dtype=bool)
    if state.always_dead_rows.size:
        dead[state.always_dead_rows] = True
    if state.direct_axon.size:
        dead[state.direct_axon] = col_dead[state.direct_producer]
    if state.group_axon.size:
        live = ~col_dead[state.group_producers]
        live_per_group = np.add.reduceat(
            live.astype(np.int64), state.group_offsets[:-1]
        )
        dead[state.group_axon] = live_per_group == 0
    return dead


def flat_orphan_neurons(state: FlatState, row_dead: np.ndarray) -> np.ndarray:
    """Twin of ``_orphans_from_plan`` over the whole graph at once.

    A port with no consumer is orphaned outright; otherwise it dies when every
    consuming axon row is dead. Output and transfer-protected ports never
    orphan. Returns a boolean over flat neuron ports.
    """
    dead = np.zeros(state.n_cols, dtype=bool)
    if state.orphan_now.size:
        dead[state.orphan_now] = True
    if state.consumer_port.size:
        live = ~row_dead[state.consumer_rows]
        live_per_port = np.add.reduceat(
            live.astype(np.int64), state.consumer_offsets[:-1]
        )
        dead[state.consumer_port] = live_per_port == 0
    dead[state.never_orphan] = False
    return dead
