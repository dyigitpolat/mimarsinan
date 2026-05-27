"""Hybrid segmented TTFS must match flat IR oracle on pruned graphs with ComputeOp boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.spiking.unified.flow import SpikingUnifiedCoreFlow

SIM_LENGTH = 8
CORES_CONFIG = [{"count": 64, "max_axons": 32, "max_neurons": 32}]


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


def _build_pruned_two_segment_graph():
    """Two neural cores, mean ComputeOp boundary, partial column pruning on core B."""
    a = NeuralCore(
        id=0,
        name="A",
        input_sources=_src([(-2, 0), (-2, 1)]),
        core_matrix=np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float64),
        hardware_bias=np.zeros(2, dtype=np.float64),
        threshold=1.0,
        latency=0,
    )
    bridge = ComputeOp(
        id=2,
        name="bridge",
        op_type="Identity",
        input_sources=_src([(0, 0), (0, 1)]),
        params={"module": nn.Identity()},
        output_shape=(1, 2),
    )
    b = NeuralCore(
        id=1,
        name="B",
        input_sources=_src([(2, 0), (2, 1), (-1, 0)]),
        core_matrix=np.array([[0.5, 0.0], [0.0, 0.0]], dtype=np.float64),
        hardware_bias=np.array([0.2, 0.0], dtype=np.float64),
        threshold=1.0,
        latency=1,
    )
    graph = IRGraph(
        nodes=[a, bridge, b],
        output_sources=_src([(1, 0), (1, 1)]),
    )
    graph = prune_ir_graph(
        graph,
        spiking_mode="ttfs",
        simulation_steps=SIM_LENGTH,
    )
    IRLatency(graph).calculate()
    return graph


@pytest.mark.parametrize("spiking_mode", ["ttfs", "ttfs_quantized"])
def test_hybrid_matches_unified_on_pruned_computeop_graph(spiking_mode):
    ir_graph = _build_pruned_two_segment_graph()
    input_shape = (1, 2)
    hybrid_mapping = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=CORES_CONFIG,
    )
    assert any(s.kind == "compute" for s in hybrid_mapping.stages)
    assert len(hybrid_mapping.stages) >= 2

    soft = SpikingUnifiedCoreFlow(
        input_shape, ir_graph, SIM_LENGTH, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode=spiking_mode,
    ).eval()
    hard = SpikingHybridCoreFlow(
        input_shape, hybrid_mapping, SIM_LENGTH, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode=spiking_mode,
    ).eval()

    x = torch.rand(4, *input_shape)
    with torch.no_grad():
        soft_out = soft(x)
        hard_out = hard(x) / float(SIM_LENGTH)

    max_diff = (soft_out - hard_out).abs().max().item()
    assert max_diff < 1e-6, (
        f"{spiking_mode}: hybrid vs unified max diff {max_diff:.3e}. "
        f"soft={soft_out[0]}, hard={hard_out[0]}"
    )
