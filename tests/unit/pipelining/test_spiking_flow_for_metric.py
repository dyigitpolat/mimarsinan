"""Metric flow uses hybrid mapping execution for all spiking modes."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.simulation_factory import build_spiking_hybrid_flow


@pytest.mark.parametrize(
    "spiking_mode, firing_mode, spike_gen_mode",
    [
        ("ttfs", "TTFS", "TTFS"),
        ("ttfs_quantized", "TTFS", "TTFS"),
        ("lif", "Default", "Uniform"),
    ],
)
def test_build_spiking_hybrid_flow_routing(spiking_mode, firing_mode, spike_gen_mode):
    pipeline = MagicMock()
    pipeline.config = {
        "spiking_mode": spiking_mode,
        "input_shape": (1, 28, 28),
        "simulation_steps": 4,
        "firing_mode": firing_mode,
        "spike_generation_mode": spike_gen_mode,
        "thresholding_mode": "<=",
        "device": "cpu",
        "cycle_accurate_lif_forward": False,
    }
    flow = build_spiking_hybrid_flow(pipeline, hybrid_mapping=MagicMock())
    assert isinstance(flow, SpikingHybridCoreFlow)
