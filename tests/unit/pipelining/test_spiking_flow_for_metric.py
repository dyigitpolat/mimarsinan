"""Metric flow uses hybrid mapping execution for all spiking modes."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.simulation_factory import build_spiking_flow_for_metric


@pytest.mark.parametrize("spiking_mode", ["ttfs", "ttfs_quantized", "lif"])
def test_build_spiking_flow_for_metric_routing(spiking_mode):
    pipeline = MagicMock()
    pipeline.config = {
        "spiking_mode": spiking_mode,
        "input_shape": (1, 28, 28),
        "simulation_steps": 4,
        "firing_mode": "TTFS",
        "spike_generation_mode": "TTFS",
        "thresholding_mode": "<=",
        "device": "cpu",
        "cycle_accurate_lif_forward": False,
    }
    ir_graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
    flow = build_spiking_flow_for_metric(
        pipeline, hybrid_mapping=MagicMock(), ir_graph=ir_graph,
    )
    assert isinstance(flow, SpikingHybridCoreFlow)
