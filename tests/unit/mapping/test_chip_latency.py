"""Regression tests for :class:`ChipLatency`.

Pins the contract the rest of the HCM relies on: every core's
``latency`` must equal the depth at which its outputs are *needed* by
its consumers, so the per-cycle buffer cascade in
``SpikingHybridCoreFlow._run_neural_segment_rate`` delivers each
source's full active window to each consumer (and the SANA-FE parity
recording in ``SanafeRunner._derive_per_core_input_counts`` agrees).

The pre-fix bug: when IR pruning rewires every axon source of a core
to ``off`` (Phase 2 of ``prune_ir_graph``), the backward walk treats
that core as ``latency=0`` because off-sources return delay 0.
Consumers at a real depth ``L`` then read its single-cycle buffer at
``[L, L+T)`` while the source only updates the buffer during ``[0, T)``,
so the consumer integrates the stale cycle ``T-1`` firing ``T`` times
and SANA-FE's spike trace count diverges from HCM's ``record_in_t``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.chip_latency import ChipLatency


def _off() -> SpikeSource:
    return SpikeSource(-1, 0, False, True)  # is_off_ = True


def _on() -> SpikeSource:
    return SpikeSource(-1, 0, True, False)  # is_always_on_ = True


def _input(idx: int) -> SpikeSource:
    return SpikeSource(-2, idx, False, False)  # is_input_ = True


def _xcore(core_idx: int, neuron: int) -> SpikeSource:
    return SpikeSource(core_idx, neuron, False, False)


def _make_core(axon_sources, weights):
    """Build a minimal core stub that ChipLatency can walk."""
    weights = np.asarray(weights, dtype=np.float32)
    return SimpleNamespace(
        axon_sources=list(axon_sources),
        core_matrix=weights,
        latency=None,
    )


def _make_mapping(cores, output_sources):
    return SimpleNamespace(cores=cores, output_sources=list(output_sources))


def test_dead_input_core_aligns_with_consumer_depth():
    """A core whose every live axon is ``off`` must inherit the depth
    its consumers need, not 0 — so the cascade buffer arrives at the
    consumer's read cycle.
    """
    # Core 0: a "real" chain depth-0 (segment input → core 0).
    # Core 1, 2, 3, 4, 5: cascade depth-1 .. depth-5.
    # Core 6 (consumer at depth 5): reads from core 5 *and* from a
    #   dead-input core (core 7), whose every axon source is off.
    # Core 7 (dead-input): no live cross-core source.
    # Backward walk gives core 7 latency 0; the fix bumps it to 4 so
    # its T-cycle window ends exactly when core 6 finishes integrating.
    c0 = _make_core([_input(0), _input(1)], [[1.0], [1.0]])
    c1 = _make_core([_xcore(0, 0)], [[1.0]])
    c2 = _make_core([_xcore(1, 0)], [[1.0]])
    c3 = _make_core([_xcore(2, 0)], [[1.0]])
    c4 = _make_core([_xcore(3, 0)], [[1.0]])
    c5 = _make_core([_xcore(4, 0)], [[1.0]])
    c7 = _make_core([_off(), _off()], [[1.0], [1.0]])  # dead-input
    c6 = _make_core([_xcore(5, 0), _xcore(7, 0)], [[1.0], [1.0]])

    cores = [c0, c1, c2, c3, c4, c5, c6, c7]
    output_sources = [_xcore(6, 0)]
    mapping = _make_mapping(cores, output_sources)

    max_lat = ChipLatency(mapping).calculate()

    assert cores[0].latency == 0
    assert cores[1].latency == 1
    assert cores[5].latency == 5
    assert cores[6].latency == 6  # consumer = 1 + deepest source's lat
    # The bug: c7 used to land at 0 here. The fix lifts it to c6.lat - 1 = 5
    # so c7's active window ends just as c6 starts integrating.
    assert cores[7].latency == 5, (
        f"dead-input core should be bumped to consumer.lat - 1 = 5; got {cores[7].latency}"
    )
    assert max_lat == max(c.latency + 1 for c in cores)


def test_dead_input_chain_propagates_alignment_upstream():
    """When the only path to a real consumer goes through a chain of
    dead-input cores, every link in the chain should be bumped so the
    cascade is consistent.
    """
    # c_dead0 (dead-input) → c_dead1 → c_real_consumer @ depth 6.
    # Both dead-side cores are "shiftable": c_dead0 has no live source,
    # c_dead1's only live source is c_dead0 (which is shiftable). So
    # both should bump: c_dead1 → 5, c_dead0 → 4.
    real_chain = []
    # build a real depth-6 chain on the side so the consumer's latency
    # is fixed at 6 regardless of the dead chain's contribution.
    prev = _make_core([_input(0)], [[1.0]])
    real_chain.append(prev)
    for _ in range(5):
        nxt = _make_core([_xcore(len(real_chain) - 1, 0)], [[1.0]])
        real_chain.append(nxt)
    # real_chain has depths 0..5 at indices 0..5.

    c_dead0 = _make_core([_off()], [[1.0]])
    # c_dead1 reads c_dead0 (which is shiftable).
    c_dead1 = _make_core([_xcore(len(real_chain), 0)], [[1.0]])  # idx will be len(real_chain)
    # Consumer reads both ends.
    consumer = _make_core(
        [_xcore(len(real_chain) - 1, 0), _xcore(len(real_chain) + 1, 0)],
        [[1.0], [1.0]],
    )

    cores = real_chain + [c_dead0, c_dead1, consumer]
    consumer_idx = len(cores) - 1
    mapping = _make_mapping(cores, [_xcore(consumer_idx, 0)])

    ChipLatency(mapping).calculate()

    # Consumer latency is 6 (one above the real chain's depth-5 tail).
    assert cores[consumer_idx].latency == 6
    # The dead chain must align: c_dead1 → 5, c_dead0 → 4.
    c_dead0_idx = len(real_chain)
    c_dead1_idx = len(real_chain) + 1
    assert cores[c_dead1_idx].latency == 5
    assert cores[c_dead0_idx].latency == 4


def test_always_on_input_axons_count_as_shiftable_base():
    """An always-on axon is a deterministic per-cycle source; a core
    whose only live axons are always-on (and otherwise off) is still
    bias-/constant-driven and should align with its consumer's depth.
    """
    # Depth-0..3 real chain.
    c0 = _make_core([_input(0)], [[1.0]])
    c1 = _make_core([_xcore(0, 0)], [[1.0]])
    c2 = _make_core([_xcore(1, 0)], [[1.0]])
    c3 = _make_core([_xcore(2, 0)], [[1.0]])
    # Bias-only core: just an always-on axon (no cross-core input).
    c_bias = _make_core([_on()], [[1.0]])
    # Consumer at depth 3 reads from c2 (lat=2 source) and c_bias.
    consumer = _make_core([_xcore(2, 0), _xcore(4, 0)], [[1.0], [1.0]])

    cores = [c0, c1, c2, c3, c_bias, consumer]
    mapping = _make_mapping(cores, [_xcore(5, 0)])
    ChipLatency(mapping).calculate()

    # Consumer latency is 3 (one above its deepest real source at 2).
    assert cores[5].latency == 3
    # c_bias has no live cross-core source → must bump to 2.
    assert cores[4].latency == 2


def test_no_shiftable_cores_leaves_backward_walk_intact():
    """Sanity: when every core has at least one live cross-core input
    (no dead-input situation), the post-pass leaves the backward walk
    untouched."""
    c0 = _make_core([_input(0)], [[1.0]])
    c1 = _make_core([_xcore(0, 0)], [[1.0]])
    c2 = _make_core([_xcore(1, 0)], [[1.0]])
    cores = [c0, c1, c2]
    mapping = _make_mapping(cores, [_xcore(2, 0)])
    ChipLatency(mapping).calculate()

    assert cores[0].latency == 0
    assert cores[1].latency == 1
    assert cores[2].latency == 2
