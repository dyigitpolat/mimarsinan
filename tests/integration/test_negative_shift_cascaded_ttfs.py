"""Cascaded ttfs_cycle_based inherits the rate-path negative shift (HCM side)."""

from __future__ import annotations

import numpy as np
import torch

from integration.parity_harness import build_toy_hybrid_mapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow


def _cascade_flow(hybrid, T=4):
    import torch.nn as nn

    return SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode="ttfs_cycle_based",
        ttfs_cycle_schedule="cascaded",
    )


def test_cascade_apply_input_shifts_adds_per_slice():
    hybrid = build_toy_hybrid_mapping()
    hybrid.node_output_shifts = {-2: np.array([0.25])}
    flow = _cascade_flow(hybrid)
    stage = hybrid.stages[0]
    rates = torch.tensor([[-0.25]], dtype=torch.float64)
    out = flow._apply_input_shifts(stage.input_map, rates)
    torch.testing.assert_close(out, torch.tensor([[0.0]], dtype=torch.float64))


def test_cascade_negative_input_recovered_by_shift():
    """A negative boundary rate is clamped to silence without the shift; with the
    producer shift the cascade core fires (TTFS-encoded recovered value)."""
    T = 8
    value = -0.5

    hybrid_n = build_toy_hybrid_mapping()
    flow_n = _cascade_flow(hybrid_n, T)
    with torch.no_grad():
        out_n = flow_n(torch.tensor([[value]], dtype=torch.float64))

    hybrid_s = build_toy_hybrid_mapping()
    hybrid_s.node_output_shifts = {-2: np.array([1.0])}
    flow_s = _cascade_flow(hybrid_s, T)
    with torch.no_grad():
        out_s = flow_s(torch.tensor([[value]], dtype=torch.float64))

    assert float(out_n.reshape(-1)[0]) == 0.0
    assert float(out_s.reshape(-1)[0]) > 0.0
