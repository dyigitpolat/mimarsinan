"""A ComputeOp's consumer decref walks its producers, not its every axon.

Measured (real 4,925-core ViT, 2026-08-07): the host ops carry up to 605,184
input sources each, and every forward re-walked all of them to decrement the
state-buffer refcounts — roughly 23M Python iterations per forward, shared by
the value, rate and TTFS flows alike.

The multiset is static, so it is counted ONCE per op; decrementing a producer
by its occurrence count in one step is arithmetically identical to
decrementing it one occurrence at a time (nothing observes the intermediate
states within a single op's decref).
"""

import numpy as np
import pytest

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    decref_consumers,
    decref_op_consumers,
    op_source_counts,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource


def _op(pairs, op_id=1):
    return ComputeOp(
        id=op_id, name="op",
        input_sources=np.array(
            [IRSource(node_id=n, index=i) for n, i in pairs], dtype=object,
        ),
        op_type="Identity", params={}, input_shape=(len(pairs),),
        output_shape=(len(pairs),),
    )


class TestCountsMatchTheSources:
    def test_counts_are_the_occurrence_multiset(self):
        op = _op([(3, 0), (3, 1), (5, 0), (-1, 0), (-2, 0), (3, 2)])
        assert op_source_counts(op) == {3: 3, 5: 1}, "negatives are not producers"

    def test_counts_are_cached_not_recomputed(self):
        op = _op([(7, i) for i in range(50)])
        first = op_source_counts(op)
        assert op_source_counts(op) is first


class TestBulkDecrefEqualsOneAtATime:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_identical_state_after(self, seed):
        rng = np.random.default_rng(seed)
        pairs = [(int(rng.integers(-2, 6)), int(rng.integers(0, 4)))
                 for _ in range(60)]
        op = _op(pairs)
        producers = [n for n, _ in pairs if n >= 0]
        base = {}
        for n in producers:
            base[n] = base.get(n, 0) + 1
        # give some producers extra outstanding reads, some exactly enough
        remaining_a = {n: c + int(rng.integers(0, 3)) for n, c in base.items()}
        remaining_b = dict(remaining_a)
        buf_a = {n: f"tensor{n}" for n in base}
        buf_b = dict(buf_a)

        decref_consumers(buf_a, remaining_a, list(producers))
        decref_op_consumers(buf_b, remaining_b, op)

        assert remaining_b == remaining_a, "refcounts diverged"
        assert buf_b == buf_a, "state-buffer eviction diverged"

    def test_unknown_producers_are_ignored_the_same_way(self):
        op = _op([(9, 0), (9, 1)])
        buf_a, buf_b = {}, {}
        rem_a, rem_b = {}, {}
        decref_consumers(buf_a, rem_a, [9, 9])
        decref_op_consumers(buf_b, rem_b, op)
        assert rem_b == rem_a and buf_b == buf_a

    def test_spike_buffer_shares_the_lifetime(self):
        op = _op([(4, 0)])
        buf_a, spikes_a = {4: "t"}, {4: "s"}
        buf_b, spikes_b = {4: "t"}, {4: "s"}
        decref_consumers(buf_a, {4: 1}, [4], state_buffer_spikes=spikes_a)
        decref_op_consumers(buf_b, {4: 1}, op, state_buffer_spikes=spikes_b)
        assert buf_b == buf_a and spikes_b == spikes_a == {}


class TestTheWalkIsNotPerSource:
    def test_decref_cost_does_not_scale_with_source_count(self):
        """The regression this closes: 605K sources walked per op per forward."""
        wide = _op([(2, i) for i in range(20000)])
        counts = op_source_counts(wide)
        assert len(counts) == 1, "one producer, however many axons read it"
        remaining = {2: 20000}
        decref_op_consumers({2: "t"}, remaining, wide)
        assert remaining == {}, "one bulk decrement must retire the producer"
