"""[U-LIVE] The class by a generator: a dead wired axon counts for no scheduling rule.

The window unit's generated cascades with dead in-segment reads (``cascade_trees`` with ``dead``: input columns held at
zero from a producer other cores still read, a producer read only through a dead read, a chain of such producers) are
mapped as a program built outside Soft Core Mapping (no dead-read pass, no relay), in the count family under both LIF
semantics and in the timing family, and read twice: as drawn, every dead read wired, and with every all-zero row's
source turned off by this test's own reading (nothing else changed). A dead axon delivers nothing, so every
scheduling rule must read one schedule from both: IRLatency's tiers, the relay planner's gaps, the window rule's IR
gaps, the static check's identity timing, and on the identity program ChipLatency's latencies and the executor's window
gaps, core by core (a program built without relays keeps its live gaps, the same on both sides). On the window unit's
tree ChipLatency's sweep, its alignment, the IR's tiers, the window rule's
latencies and the static check counted the wired reads.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import unit.spiking.cascade_trees as ct
import unit.spiking.cascade_windows as cw
import unit.spiking.layout_trees as lt

from mimarsinan.certification.program_check.unbounded.latency import identity_timing
from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.latency.cycle_windows import core_window_gaps, ir_window_gaps
from mimarsinan.mapping.latency.depth_balancing import find_intra_segment_depth_gaps
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

TREES = 60
FAMILIES = [("count", False), ("count", True), ("timing", False)]


def _ir(seed: int, family: str):
    tree = ct.draw_cascade(seed, True)
    flow = lt.GraphFlow(tree.repr_).eval()
    with torch.no_grad():
        flow(torch.rand(2, *tree.shape))
    gauges = [torch.as_tensor(g, dtype=torch.float32) for g in tree.gauges]
    if family == "timing":
        repr_, _ = cw.timing_activations(flow, gauges, S=ct.TIMING_S, perceptrons=tree.perceptrons)
        return IRMapping(q_max=127.0, firing_mode="TTFS", max_axons=2048, max_neurons=2048).map(repr_)
    mark_encoding_layers(flow.get_mapper_repr())
    for p, g in zip(tree.perceptrons, gauges):
        p.set_activation_scale(g.clone())
        lif = LIFActivation(T=8, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    repr_ = flow.get_mapper_repr()
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    repr_.assign_perceptron_indices()
    return IRMapping(q_max=127.0, firing_mode="Default", max_axons=256, max_neurons=256).map(repr_)


def _reads_off(ir):
    """This test's own reading: every source whose weight row is all zero, turned off."""
    out = copy.deepcopy(ir)
    turned = 0
    for node in out.nodes:
        if not isinstance(node, NeuralCore):
            continue
        matrix = np.asarray(node.get_core_matrix(out))
        flat = node.input_sources.flatten()
        if matrix.ndim != 2 or matrix.shape[0] != len(flat):
            continue
        for i, s in enumerate(flat):
            if isinstance(s, IRSource) and s.node_id >= 0 and not np.any(matrix[i] != 0):
                flat[i] = IRSource(node_id=-1, index=0)
                turned += 1
        node.input_sources = flat.reshape(node.input_sources.shape)
    return out, turned


def _schedule(ir, per_hop: bool) -> dict:
    ir = copy.deepcopy(ir)
    tiers = IRLatency(ir).calculate()
    out = {"tiers": (tiers, [n.latency for n in ir.nodes if isinstance(n, NeuralCore)]),
           "relay_gaps": sorted((g.producer_id, g.consumer_id, g.gap) for g in find_intra_segment_depth_gaps(ir)),
           "window_gaps": sorted(ir_window_gaps(ir, per_hop=per_hop)),
           "identity_timing": identity_timing(ir)[0]}
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=per_hop)
    stages = []
    for stage in hybrid.stages:
        hcm = getattr(stage, "hard_core_mapping", None)
        if getattr(stage, "kind", None) != "neural" or hcm is None:
            continue
        ChipLatency(hcm).calculate()
        stages.append(([int(c.latency or 0) for c in hcm.cores], sorted(core_window_gaps(hcm))))
    out["chip"] = stages
    return out


@pytest.mark.parametrize("family,per_hop", FAMILIES, ids=["count_streamed", "count_windowed", "timing"])
def test_a_dead_read_counts_for_no_scheduling_rule(family, per_hop):
    apart, drawn = [], 0
    for seed in range(TREES):
        try:
            wired = _ir(seed, family)
        except Exception as exc:  # noqa: BLE001 - a construction the mapping declines by name draws nothing here
            if lt.refused_by_name(exc):
                continue
            raise
        reference, turned = _reads_off(wired)
        drawn += int(turned > 0)
        a, b = _schedule(wired, per_hop), _schedule(reference, per_hop)
        rules = [k for k in a if a[k] != b[k]]
        if rules:
            apart.append((seed, rules))
    assert not apart, apart[:5]
    assert drawn >= TREES // 2, drawn  # the generator draws the class
