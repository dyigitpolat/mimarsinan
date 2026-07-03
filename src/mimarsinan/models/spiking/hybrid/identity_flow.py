"""Identity-mapped SpikingHybridCoreFlow builder (rung-2 executor, no pipeline)."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.support.bias_compensation import (
    propagate_negative_shifts_to_hybrid,
)
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow


def build_identity_spiking_flow(
    input_shape,
    ir_graph,
    simulation_length: int,
    preprocessor: nn.Module | None = None,
    firing_mode: str = "Default",
    spike_mode: str = "Uniform",
    thresholding_mode: str = "<=",
    *,
    spiking_mode: str = "lif",
    ttfs_cycle_schedule: str = "cascaded",
) -> SpikingHybridCoreFlow:
    """Run an IRGraph through the hybrid executor on a 1:1 identity mapping.

    Latencies are computed only when the graph has none preset.
    """
    if any(
        getattr(node, "latency", None) is None
        for node in ir_graph.get_neural_cores()
    ):
        IRLatency(ir_graph).calculate()

    mapping = build_identity_hybrid_mapping(ir_graph=ir_graph)
    propagate_negative_shifts_to_hybrid(ir_graph, mapping)
    return SpikingHybridCoreFlow(
        input_shape,
        mapping,
        simulation_length,
        preprocessor,
        firing_mode,
        spike_mode,
        thresholding_mode,
        spiking_mode=spiking_mode,
        ttfs_cycle_schedule=ttfs_cycle_schedule,
    )
