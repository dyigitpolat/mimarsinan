"""build_identity_spiking_flow: rung-2 executor over an identity mapping."""

import numpy as np
import torch
import torch.nn as nn

from conftest import make_tiny_ir_graph

from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow


def test_builds_hybrid_flow_over_identity_mapping():
    ir_graph = make_tiny_ir_graph()
    flow = build_identity_spiking_flow(
        (8,), ir_graph, 4, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs_quantized",
    )
    assert isinstance(flow, SpikingHybridCoreFlow)
    for stage in flow.hybrid_mapping.stages:
        if stage.kind != "neural":
            continue
        for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
            assert len(placements) == 1


def test_respects_preset_latencies():
    ir_graph = make_tiny_ir_graph()  # conftest presets latencies 0 and 1
    preset = [n.latency for n in ir_graph.get_neural_cores()]
    build_identity_spiking_flow(
        (8,), ir_graph, 4, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs_quantized",
    )
    assert [n.latency for n in ir_graph.get_neural_cores()] == preset


def test_computes_latencies_when_missing():
    ir_graph = make_tiny_ir_graph()
    for n in ir_graph.get_neural_cores():
        n.latency = None
    build_identity_spiking_flow(
        (8,), ir_graph, 4, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs_quantized",
    )
    assert all(n.latency is not None for n in ir_graph.get_neural_cores())


def test_forward_runs_and_is_finite():
    torch.manual_seed(0)
    ir_graph = make_tiny_ir_graph()
    flow = build_identity_spiking_flow(
        (8,), ir_graph, 4, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode="ttfs_quantized",
    ).eval()
    with torch.no_grad():
        out = flow(torch.rand(2, 8))
    assert out.shape == (2, 4)
    assert np.isfinite(out.numpy()).all()
