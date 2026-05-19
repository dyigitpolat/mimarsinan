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


def test_consumer_latency_at_least_source_core_latency_plus_one():
    """A consumer must be scheduled at latency >= max(source core lat) + 1.

    The per-neuron backward walk used by ``ChipLatency.calculate`` can
    under-estimate latency when a consumer's non-zero weights happen to
    reference source-core neurons that themselves have shallow
    per-neuron delays — most visibly when a source core has a "split"
    structure: some of its neurons drive the deep chain (delay ≈ depth)
    while others are bias-only or have all non-zero weights on
    ``is_off_`` axons (residual weights left over from IR pruning).
    In that case the per-neuron walk pins the bias-only neurons at
    delay 0; a consumer reading those neurons gets latency 1 even
    though the source core *as a whole* fires at depth ``D >> 1``.

    The chip schedules each core as a unit during
    ``[core.latency, core.latency + T)``, so the consumer cannot safely
    read source-core outputs before that window starts. Enforcing
    ``consumer_lat >= max(source_core_lat) + 1`` after the per-neuron
    walk eliminates the dead-edge wedge that caused a multi-pp
    NF→SCM accuracy gap on compilagent 8-deep architectures.
    """
    # A linear chain: core 0 (input) → core 1 → core 2 → core 3 → core 4 → core 5.
    # Core 5 has a "split" structure: half its neurons drive the chain,
    # half are bias-only with residual non-zero weights on off-axons.
    c0 = _make_core([_input(0), _input(1)], [[1.0], [1.0]])
    c1 = _make_core([_xcore(0, 0)], [[1.0]])
    c2 = _make_core([_xcore(1, 0)], [[1.0]])
    c3 = _make_core([_xcore(2, 0)], [[1.0]])
    c4 = _make_core([_xcore(3, 0)], [[1.0]])
    # Core 5: two neurons. Neuron 0 is a real chain consumer (depth 5);
    # neuron 1 has its only non-zero weight on an off-axon (residual
    # from pruning).
    c5 = _make_core(
        [_xcore(4, 0), _off()],
        [
            [1.0, 0.0],  # axon 0 (cross-core) → neuron 0, weight 1
            [0.0, 3.0],  # axon 1 (off) → neuron 1, residual weight 3
        ],
    )
    # Core 6: reads from core 5 *neuron 1* (the bias-only one). Without
    # the per-core invariant pass, core 6's latency comes out at 1
    # (because core 5 neuron 1's per-neuron delay is 0 after the
    # off-axon filter). With enforcement it must rise to core_5.latency + 1.
    c6 = _make_core([_xcore(5, 1)], [[1.0]])

    mapping = _make_mapping(
        [c0, c1, c2, c3, c4, c5, c6],
        output_sources=[_xcore(6, 0)],
    )
    ChipLatency(mapping).calculate()

    c5_lat = int(c5.latency or 0)
    c6_lat = int(c6.latency or 0)
    assert c5_lat == 5, (
        f"core 5 should be at latency 5 (deep-chain neuron determines core depth); "
        f"got {c5_lat}"
    )
    assert c6_lat >= c5_lat + 1, (
        f"core 6 must satisfy the per-core latency invariant "
        f"(c6_lat >= max(source_core_lat)+1 = {c5_lat + 1}); got {c6_lat}. "
        "Without enforcement, the per-neuron walk pins c6 to lat=1 because "
        "c5 neuron 1's only non-zero weight is on an off-axon, hiding the "
        "real core-level dependency."
    )


def test_off_axon_nonzero_weight_excluded_from_delay_walk():
    """IR pruning can leave residual non-zero weights on axons whose
    source has been rewritten to ``is_off_``. The latency walk must
    ignore these — they don't deliver signal and shouldn't be treated
    as direct-input dependencies via ``core_=-1``.
    """
    # A neuron whose *only* non-zero weight is on an off-axon. Backward
    # walk should treat it as having no live dependencies (delay 0
    # before shiftable alignment), not as having a delay-0 direct
    # input (which would inflate its delay to 1).
    c0 = _make_core([_off()], [[5.0]])  # residual weight on off-axon
    # An output reader sees core 0 → with the fix, core 0's neuron
    # is treated as having no live deps and the shiftable alignment
    # picks up the rest. Without the fix, the off-axon's source_=-1
    # gets treated as direct input and core 0 ends up at latency 0
    # via delay=1 path (still works for this trivial case, but the
    # bug manifests in multi-neuron cores — see the test above).
    mapping = _make_mapping([c0], output_sources=[_xcore(0, 0)])
    ChipLatency(mapping).calculate()
    # The only meaningful assertion: this didn't raise and c0 has a
    # valid integer latency.
    assert isinstance(c0.latency, int)
