"""Round-2a flow wiring: per-producer-channel shift applied before the [0,1] clamp.

`node_output_shifts` on the hybrid mapping makes a negative-producing boundary
lossless for spike encoding (the consumer bias is pre-corrected separately, see
`test_neg_shift_bias`). This pins the flow mechanism: the shift is added to the
right input slices before clamping, and is the identity when the table is empty
(default ⇒ zero behavior change).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from integration.parity_harness import build_toy_hybrid_mapping, default_behavior


def _flow(hybrid, T=4):
    b = default_behavior()
    return SpikingHybridCoreFlow(
        input_shape=(1,), hybrid_mapping=hybrid, simulation_length=T,
        preprocessor=nn.Identity(), firing_mode=b.firing_mode,
        spike_mode=b.spike_generation_mode, thresholding_mode=b.thresholding_mode,
        spiking_mode=b.spiking_mode,
    ).eval()


def test_apply_input_shifts_adds_per_slice():
    hybrid = build_toy_hybrid_mapping()
    hybrid.node_output_shifts = {-2: np.array([0.3])}
    flow = _flow(hybrid)
    stage = hybrid.stages[0]
    rates = torch.tensor([[0.5]], dtype=torch.float64)
    shifted = flow._apply_input_shifts(stage.input_map, rates)
    torch.testing.assert_close(shifted, torch.tensor([[0.8]], dtype=torch.float64))
    # Original tensor must not be mutated (shift clones).
    torch.testing.assert_close(rates, torch.tensor([[0.5]], dtype=torch.float64))


def test_apply_input_shifts_identity_when_empty():
    hybrid = build_toy_hybrid_mapping()  # node_output_shifts defaults to {}
    flow = _flow(hybrid)
    stage = hybrid.stages[0]
    rates = torch.tensor([[0.5]], dtype=torch.float64)
    out = flow._apply_input_shifts(stage.input_map, rates)
    assert out is rates  # identity object: no copy, no behavior change


def test_negative_input_recovered_by_shift():
    """A negative boundary value is clamped to 0 (lost) without a shift; the shift
    lifts it into [0,1] so it encodes to a non-zero spike rate."""
    hybrid = build_toy_hybrid_mapping()
    flow = _flow(hybrid)
    stage = hybrid.stages[0]
    neg = torch.tensor([[-0.4]], dtype=torch.float64)

    lost = flow._apply_input_shifts(stage.input_map, neg).clamp(0.0, 1.0)
    assert float(lost) == 0.0  # without shift: clamped away

    hybrid.node_output_shifts = {-2: np.array([0.4])}
    recovered = flow._apply_input_shifts(stage.input_map, neg).clamp(0.0, 1.0)
    assert float(recovered) == 0.0  # -0.4 + 0.4 = 0.0 (boundary); raise shift to recover
    hybrid.node_output_shifts = {-2: np.array([0.9])}
    recovered2 = flow._apply_input_shifts(stage.input_map, neg).clamp(0.0, 1.0)
    assert abs(float(recovered2) - 0.5) < 1e-9  # -0.4 + 0.9 = 0.5, now encodable
