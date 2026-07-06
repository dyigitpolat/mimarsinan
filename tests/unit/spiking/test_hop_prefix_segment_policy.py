"""[5v B2] intra-segment hop frontier: genuine prefix hops + trained-proxy suffix.

t0_16 proved the P4 need is BELOW segments (1 segment, 9 hops, "blend fallback
by design" = the crater): the frontier unit becomes the discretization hop.
Nodes with cascade depth < k run the deployed single-spike cycle loop; deeper
nodes run the trained value proxy on the frontier's DECODED values (exactly the
study's prefix-probe k-hybrid). k=0 is the trained proxy exactly; k=n_levels is
the deployed cascade exactly; the genuine prefix is bit-invariant to the suffix
mode.
"""

from __future__ import annotations

import pytest
import torch

from cascade_fixtures import build_cascade_flow

from mimarsinan.models.spiking.training.ttfs_segment_forward import (
    PrefixTTFSSegmentForward,
    TTFSSegmentForward,
)

_S = 4
_DEPTH = 6


@pytest.fixture(scope="module")
def flow_and_x():
    flow, calib_x = build_cascade_flow(host_ops=False, depth=_DEPTH, S=_S)
    return flow, calib_x[:8]


def _forward(flow, x, *, hop_k=None, genuine=True):
    executor = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
    if hop_k is None:
        executor.set_prefix(executor.n_segments if genuine else 0)
    else:
        executor.set_hop_prefix(hop_k)
    with torch.no_grad():
        return executor(x)


def _forward_with_values(flow, x, *, hop_k=None):
    executor = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
    if hop_k is not None:
        executor.set_hop_prefix(hop_k)
    with torch.no_grad():
        return executor.forward_with_node_values(x)


class TestHopLevels:
    def test_single_segment_chain_reports_its_depth_levels(self, flow_and_x):
        flow, _ = flow_and_x
        executor = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        assert executor.n_segments == 1
        assert executor.n_hop_levels == _DEPTH


class TestEndpointEquivalence:
    def test_full_hop_frontier_is_the_deployed_cascade_bit_exactly(self, flow_and_x):
        flow, x = flow_and_x
        reference = _forward(flow, x, genuine=True)
        full = _forward(flow, x, hop_k=_DEPTH)
        assert torch.equal(full, reference)

    def test_zero_hop_frontier_is_the_trained_proxy_bit_exactly(self, flow_and_x):
        flow, x = flow_and_x
        reference = _forward(flow, x, genuine=False)
        zero = _forward(flow, x, hop_k=0)
        assert torch.equal(zero, reference)

    def test_full_frontier_collapses_the_hop_state(self, flow_and_x):
        flow, _ = flow_and_x
        executor = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        executor.set_hop_prefix(_DEPTH)
        assert executor._driver.policy.genuine_hop_frontier is None


class TestHybridComposition:
    def test_intermediate_frontier_is_a_genuine_partial_deployment(self, flow_and_x):
        # On a coarse grid the cascade and the proxy genuinely differ, so a
        # half-installed chain must sit strictly between the endpoints (an
        # inert frontier that silently equals either endpoint is a defect).
        flow, x = flow_and_x
        genuine = _forward(flow, x, hop_k=_DEPTH)
        proxy = _forward(flow, x, hop_k=0)
        assert not torch.equal(genuine, proxy), "fixture must discriminate"
        mid = _forward(flow, x, hop_k=3)
        assert not torch.equal(mid, genuine)
        assert not torch.equal(mid, proxy)

    def test_genuine_prefix_values_are_invariant_to_the_suffix_mode(self, flow_and_x):
        # The frontier hops' decoded values must be bit-equal to the fully
        # deployed cascade's — converting the suffix later never rewrites the
        # already-converted prefix's behavior (P4 soundness).
        flow, x = flow_and_x
        _, full_values = _forward_with_values(flow, x, hop_k=_DEPTH)
        _, hybrid_values = _forward_with_values(flow, x, hop_k=3)
        depths = {}
        executor = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        driver = executor._driver
        for seg_nodes in driver.segments.values():
            ordered = sorted(seg_nodes, key=lambda n: driver._index[n])
            depths.update(driver.policy.segment_depths(driver, ordered))
        prefix_nodes = [
            n for n in hybrid_values
            if n in full_values and depths.get(n, 99) < 3
        ]
        assert prefix_nodes, "the hybrid must record frontier perceptrons"
        for n in prefix_nodes:
            assert torch.equal(hybrid_values[n], full_values[n])

    def test_every_frontier_step_runs(self, flow_and_x):
        flow, x = flow_and_x
        outs = [_forward(flow, x, hop_k=k) for k in range(_DEPTH + 1)]
        for out in outs:
            assert out.shape == outs[0].shape


class TestSegmentPrefixUnchanged:
    def test_segment_frontier_path_is_untouched_by_the_hop_attr(self):
        flow, calib_x = build_cascade_flow(host_ops=True, depth=3, S=_S)
        x = calib_x[:4]
        executor = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        assert executor.n_segments > 1
        executor.set_prefix(1)
        with torch.no_grad():
            seg_out = executor(x)
        assert seg_out.shape[0] == 4
        assert executor._driver.policy.genuine_hop_frontier is None

    def test_hop_mode_asserts_single_segment(self):
        flow, _ = build_cascade_flow(host_ops=True, depth=3, S=_S)
        executor = PrefixTTFSSegmentForward(flow.get_mapper_repr(), _S)
        with pytest.raises(AssertionError, match="single-segment"):
            executor.set_hop_prefix(1)
