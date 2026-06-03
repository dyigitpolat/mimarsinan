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


def test_calibrate_segment_input_mins_and_derive_shift():
    """Calibration captures the per-producer-channel min of pre-clamp boundary
    values; negative_shifts_from_min turns it into the positive-domain shift."""
    from mimarsinan.mapping.support.neg_shift_bias import negative_shifts_from_min

    hybrid = build_toy_hybrid_mapping()
    flow = _flow(hybrid)
    # Raw input node -2 feeds the toy segment; min over the batch is -0.7.
    x = torch.tensor([[-0.4], [-0.7], [0.2]], dtype=torch.float32)

    mins = flow.calibrate_segment_input_mins(x)
    assert set(mins) == {-2}
    assert abs(float(mins[-2][0]) - (-0.7)) < 1e-6

    shifts = negative_shifts_from_min(mins)
    assert abs(float(shifts[-2][0]) - 0.7) < 1e-6
    # Accumulator is cleared after calibration.
    assert flow._segment_input_min is None


def _toy_mapping_with_bias(w, b):
    """Single 1->1 core with a hardware bias, consuming raw input node -2."""
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.packing.hybrid_types import HybridStage, SegmentIOSlice
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
    from mimarsinan.mapping.ir import IRSource

    core = HardCore(4, 4, has_bias_capability=True)
    cm = np.zeros((4, 4), dtype=np.float64)
    cm[0, 0] = w
    core.core_matrix = cm
    core.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core.available_axons = 3
    core.available_neurons = 3
    core.threshold = 1.0
    core.latency = 0
    core.hardware_bias = np.array([b], dtype=np.float64)

    seg = HardCoreMapping([])
    seg.cores = [core]
    seg.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
    stage = HybridStage(
        kind="neural", name="toy", hard_core_mapping=seg,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=1)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray([IRSource(node_id=0, index=0)], dtype=object),
    )


def test_apply_negative_value_shifts_end_to_end():
    """Full orchestration: calibrate -> derive shift -> install -> bake core bias.
    A negative boundary value is recovered, and the result is count-exact with the
    equivalent positive reference (same encoded rate + same baked bias)."""
    from mimarsinan.mapping.support.neg_shift_bias import apply_negative_value_shifts

    w, b, T = 2.0, 1.5, 4
    x_cal = torch.tensor([[-0.9], [0.1]], dtype=torch.float32)  # min -0.9 -> s = 0.9
    x_test = torch.tensor([[-0.4]], dtype=torch.float32)        # -0.4 + 0.9 = 0.5

    shifted = _flow(_toy_mapping_with_bias(w, b), T)
    shifts = apply_negative_value_shifts(shifted, x_cal)
    assert abs(float(shifts[-2][0]) - 0.9) < 1e-6

    # Equivalent positive reference: store 0.5 directly, bias already b - w*0.9.
    ref = _flow(_toy_mapping_with_bias(w, b - w * 0.9), T)
    # Clamped baseline: no shift, original bias; -0.4 clamps to 0 (info lost).
    clamped = _flow(_toy_mapping_with_bias(w, b), T)

    with torch.no_grad():
        out_shifted = shifted(x_test)
        out_ref = ref(torch.tensor([[0.5]], dtype=torch.float32))
        out_clamped = clamped(x_test)

    torch.testing.assert_close(out_shifted, out_ref, atol=0.0, rtol=0.0)  # count-exact
    assert not torch.equal(out_shifted, out_clamped)  # negative was recovered
