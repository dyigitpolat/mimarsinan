"""Prefix-hybrid TTFS policy: genuine prefix + trained-proxy suffix (T4/P4).

Every k-member must be a GENUINE partial deployment: segments with series
index < k run the deployed cycle-accurate cascade (including boundary
re-encode of their inputs), the suffix runs the trained rate/staircase proxy
on decoded values — bit-equal to the validated T4 prototype semantics
(mbh_t4_depth_law.md, worktree mimarsinan_t4_iso PrefixTtfsPolicy).
"""

from __future__ import annotations

import pytest
import torch

from cascade_fixtures import build_cascade_flow, segment_count

from mimarsinan.models.spiking.training.prefix_genuine_forward import (
    PrefixGenuineForward,
    prefix_length_for_rate,
)
from mimarsinan.models.spiking.training.ttfs_segment_forward import (
    PrefixTTFSSegmentForward,
    TTFSSegmentForward,
)
from mimarsinan.spiking.segment_forward import (
    SegmentForwardDriver,
    TtfsSegmentPolicy,
    segment_series_roots,
)
from mimarsinan.spiking.segment_partition import perceptron_of

_S = 4


class _PrototypePrefixPolicy(TtfsSegmentPolicy):
    """Verbatim replica of the T4 scratch ``PrefixTtfsPolicy`` (the golden ref)."""

    def __init__(self, genuine_set):
        self._proto_genuine = set(genuine_set)
        self._seg_index: dict = {}

    def bind_driver(self, driver):
        roots = sorted(
            driver.segments.keys(),
            key=lambda r: min(driver._index[n] for n in driver.segments[r]))
        self._seg_index = {r: i for i, r in enumerate(roots)}
        return self

    def run_segment(self, driver, seg_nodes, values, x):
        root = driver._seg_of[seg_nodes[0]]
        if self._seg_index[root] in self._proto_genuine:
            return self._genuine(driver, seg_nodes, values, x)
        return self._proto_value_mode(driver, seg_nodes, values, x)

    def _genuine(self, driver, seg_nodes, values, x):
        saved = self.genuine_segments
        self.genuine_segments = None  # force the production genuine path
        try:
            return super().run_segment(driver, seg_nodes, values, x)
        finally:
            self.genuine_segments = saved

    def _proto_value_mode(self, driver, seg_nodes, values, x):
        seg_set = set(seg_nodes)
        mods = self._ttfs_nodes(seg_nodes)
        for m in mods:
            m.set_cycle_accurate(False)
        try:
            vmode = {}
            for n in seg_nodes:
                d = driver._deps.get(n, [])
                if len(d) == 1:
                    inp = vmode[d[0]] if d[0] in seg_set else values[d[0]]
                elif len(d) == 0:
                    inp = x
                else:
                    inp = tuple(
                        vmode[dep] if dep in seg_set else values[dep] for dep in d)
                vmode[n] = n.forward(inp)
        finally:
            for m in mods:
                m.set_cycle_accurate(True)
                m.reset_state()
        for n in driver.external_consumed(seg_nodes):
            values[n] = vmode[n]
        if driver._output in seg_set:
            return vmode[driver._output]
        return None


def _multi_segment_flow(depth=4, seed=0):
    flow, x = build_cascade_flow(host_ops=True, depth=depth, S=_S, seed=seed)
    return flow, x


def _prototype_forward(flow, x, k):
    policy = _PrototypePrefixPolicy(set(range(k)))
    driver = SegmentForwardDriver(flow.get_mapper_repr(), _S, policy)
    policy.bind_driver(driver)
    with torch.no_grad():
        return driver(x)


class TestPrefixBitEquality:
    """Production genuine_segments prefix == the T4 prototype, bit-equal per k."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    def test_every_k_matches_prototype(self, k):
        flow, x = _multi_segment_flow()
        n = segment_count(flow)
        assert n == 4
        expected = _prototype_forward(flow, x, k)
        fwd = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        fwd.set_prefix(k)
        with torch.no_grad():
            got = fwd(x)
        assert torch.equal(expected, got), f"k={k} diverges from the prototype"

    def test_k_equals_n_is_the_deployed_cascade(self):
        flow, x = _multi_segment_flow()
        n = segment_count(flow)
        deployed = TTFSSegmentForward(flow.get_mapper_repr(), _S)
        with torch.no_grad():
            expected = deployed(x)
        fwd = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        fwd.set_prefix(n)
        with torch.no_grad():
            got = fwd(x)
        assert torch.equal(expected, got)

    def test_default_genuine_segments_none_is_fully_genuine(self):
        # The deployed default (no prefix configured) must stay byte-identical.
        flow, x = _multi_segment_flow()
        policy = TtfsSegmentPolicy()
        assert policy.genuine_segments is None
        driver = SegmentForwardDriver(flow.get_mapper_repr(), _S, policy)
        with torch.no_grad():
            got = driver(x)
        deployed = TTFSSegmentForward(flow.get_mapper_repr(), _S)
        with torch.no_grad():
            expected = deployed(x)
        assert torch.equal(expected, got)

    def test_intermediate_k_differs_from_both_ends(self):
        flow, x = _multi_segment_flow()
        fwd = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        outs = {}
        for k in (0, 2, 4):
            fwd.set_prefix(k)
            with torch.no_grad():
                outs[k] = fwd(x)
        assert not torch.equal(outs[0], outs[2])
        assert not torch.equal(outs[2], outs[4])


class TestValueModeRecorder:
    def test_recorder_captures_all_perceptrons_across_modes(self):
        flow, x = _multi_segment_flow()
        fwd = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        fwd.set_prefix(2)
        with torch.no_grad():
            _, recorded = fwd.forward_with_node_values(x)
        recorded_ids = {id(perceptron_of(n)) for n in recorded}
        expected_ids = {id(p) for p in flow.get_perceptrons()}
        assert recorded_ids == expected_ids


class TestSuffixGradientReach:
    def test_frontier_segment_receives_gradient_through_value_suffix(self):
        flow, x = _multi_segment_flow()
        fwd = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        fwd.set_prefix(2)
        out = fwd(x)
        out.sum().backward()
        grads = []
        for p in flow.get_perceptrons():
            g = p.layer.weight.grad
            grads.append(0.0 if g is None else float(g.abs().sum()))
        # Frontier (series 1) trains through its own surrogate dynamics; the
        # value-mode suffix (series 2, 3) is differentiable; segment 0 sits
        # behind a hard boundary re-encode and is severed (the A5 reach story).
        assert grads[1] > 0.0
        assert grads[2] > 0.0 and grads[3] > 0.0
        assert grads[0] == 0.0


class TestSeriesRoots:
    def test_series_roots_are_exec_ordered(self):
        flow, _ = _multi_segment_flow()
        driver = SegmentForwardDriver(flow.get_mapper_repr(), _S, TtfsSegmentPolicy())
        roots = segment_series_roots(driver)
        firsts = [
            min(driver._index[n] for n in driver.segments[r]) for r in roots
        ]
        assert firsts == sorted(firsts)


class TestPrefixLengthForRate:
    @pytest.mark.parametrize("rate,n,expected", [
        (0.0, 8, 0),
        (1.0 / 8.0, 8, 1),
        (7.0 / 8.0, 8, 7),
        (1.0, 8, 8),
        (1.0 / 3.0, 3, 1),
        (2.0 / 3.0, 3, 2),
        (0.4999 / 8, 8, 0),
        (-0.5, 8, 0),
        (1.5, 8, 8),
    ])
    def test_ladder_rates_map_exactly(self, rate, n, expected):
        assert prefix_length_for_rate(rate, n) == expected


class TestPrefixGenuineForward:
    def test_rate_drives_prefix_live(self):
        flow, x = _multi_segment_flow()
        n = segment_count(flow)
        fwd = PrefixGenuineForward(flow, _S, rate=0.0)
        assert fwd.prefix_k == 0
        with torch.no_grad():
            proxy = fwd(x)
        fwd.rate = 1.0
        assert fwd.prefix_k == n
        with torch.no_grad():
            deployed = fwd(x)
        ref = TTFSSegmentForward(flow.get_mapper_repr(), _S)
        with torch.no_grad():
            expected = ref(x)
        assert torch.equal(deployed, expected)
        assert not torch.equal(proxy, deployed)

    def test_pickles_without_executor(self):
        import pickle

        flow, _ = _multi_segment_flow()
        fwd = PrefixGenuineForward(flow, _S, rate=0.5)
        state = pickle.loads(pickle.dumps(fwd.__getstate__()))
        assert state["_executor"] is None
        assert state["rate"] == 0.5
