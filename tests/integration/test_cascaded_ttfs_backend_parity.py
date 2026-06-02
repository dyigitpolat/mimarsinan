"""Cascaded greedy ``ttfs_cycle_based`` cross-backend parity.

The cascaded schedule is *lossy* relative to the analytical/synchronized
reference (greedy single-spike fire-once, ``S + latency`` cycles), but it must be
**internally consistent across backends**: genuine HCM (Python per-cycle cascade),
genuine nevresim (``TTFSCascadeCompute`` + ``TTFSCascadeExecution`` over single-spike
TTFS inputs) and the SANA-FE single-spike cascade soma all implement the same
single-spike fire-once + consumer-side ramp, so they agree on the decoded value.
Each neuron emits exactly ONE spike (timing = value); the per-core count is
single-spike traffic, and the decoded segment-output value is the reconstructed ramp.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

from integration.parity_harness import (
    default_behavior,
    ensure_backend_ready,
    have_sanafe,
    run_nevresim_parity,
    run_sanafe_parity,
)


def _cascaded_behavior():
    return default_behavior(
        spiking_mode="ttfs_cycle_based",
        firing_mode="TTFS",
        spike_generation_mode="TTFS",
        thresholding_mode="<=",
    )


def _build_two_core_cascade_mapping(*, w0: float, w1: float):
    """core0 (input→hidden) → core1 (hidden→output), one neural segment.

    ``w1 < threshold`` forces core1 to integrate **several** latched spikes from
    core0 before firing — directly stressing that core0's fire-once-latch emits a
    spike every cycle and that core1 accumulates them identically across backends.
    """
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.ir import IRSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        HybridHardCoreMapping, HybridStage, SegmentIOSlice,
    )
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

    core0 = HardCore(4, 4, has_bias_capability=False)
    core0.core_matrix = np.zeros((4, 4), dtype=np.float32)
    core0.core_matrix[0, 0] = w0
    core0.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core0.available_axons = 3
    core0.available_neurons = 3
    core0.threshold = 1.0
    core0.latency = 0

    core1 = HardCore(4, 4, has_bias_capability=False)
    core1.core_matrix = np.zeros((4, 4), dtype=np.float32)
    core1.core_matrix[0, 0] = w1
    core1.axon_sources = [SpikeSource(0, 0)]  # reads core0 neuron 0 (latched)
    core1.available_axons = 3
    core1.available_neurons = 3
    core1.threshold = 1.0
    core1.latency = 1

    segment = HardCoreMapping([])
    segment.cores = [core0, core1]
    segment.output_sources = np.asarray([SpikeSource(1, 0)], dtype=object)

    stage = HybridStage(
        kind="neural", name="two_core_segment", hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=1)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray([IRSource(node_id=0, index=0)], dtype=object),
    )


@pytest.mark.integration
@pytest.mark.parametrize("sample_value", [1.0, 0.625, 0.25])
def test_cascaded_ttfs_hcm_nevresim_value_parity(sample_value):
    """Genuine HCM cascade and genuine nevresim cascade agree on the decoded
    **value** (segment output), across firing times.

    Each neuron emits a *single* spike (the wire carries one event); the value is
    the ramp the consumer reconstructs (``window − fire``). So the cross-backend
    contract is the decoded segment-output value, not the raw spike count — the
    per-core ``output`` layer legitimately differs (HCM records single-spike
    *traffic* = 1; nevresim's raw is the reconstructed *value*)."""
    from mimarsinan.chip_simulation.recording.compare import format_first_diff

    ensure_backend_ready("nevresim")
    result = run_nevresim_parity(_cascaded_behavior(), T=8, sample_value=sample_value)
    assert result.nevresim_ran
    # Value layers (segment input/output) must match exactly.
    value_diffs = [d for d in result.diff_list if d.layer in ("seg_input", "seg_output")]
    assert not value_diffs, format_first_diff(value_diffs)
    # The only tolerated diffs are the per-core traffic-vs-value layers.
    for d in result.diff_list:
        assert d.layer in ("core_output", "core_input"), d.layer


@pytest.mark.integration
@pytest.mark.parametrize("sample_value", [1.0, 0.625, 0.25])
def test_cascaded_ttfs_hcm_sanafe_parity(sample_value):
    """Genuine HCM cascade and the SANA-FE fire-once-latch cascade soma agree on
    the functional cross-backend contract: per-core **output** counts and the
    gathered **segment output** are identical.

    SANA-FE's ``mimarsinan_ttfs_cascade_soma`` integrates latched TTFS inputs
    every cycle (ungated) and latches its output HIGH after the first spike —
    the same greedy dynamics as HCM/nevresim, decoded count/T. The per-core
    *input* count is the one layer that differs, by exactly the SANA-FE
    input→synapse delivery offset (the core can't integrate cycle 0's not-yet-
    delivered latched input); this is a harmless recording artifact of the
    ungated cascade — the fire-once-latch output is unchanged."""
    if not have_sanafe():
        pytest.skip("SANA-FE not installed")
    ensure_backend_ready("sanafe")
    result = run_sanafe_parity(_cascaded_behavior(), T=8, sample_value=sample_value)
    assert result.sanafe_ran
    # Functional contract: everything except the per-core input window must match.
    from mimarsinan.chip_simulation.recording.compare import format_first_diff
    functional_diffs = [d for d in result.diff_list if d.layer != "core_input"]
    assert not functional_diffs, format_first_diff(functional_diffs)
    # The only tolerated diff is the per-core input window, off by exactly the
    # +1 SANA-FE input→synapse delivery offset.
    for d in result.diff_list:
        assert d.layer == "core_input", d.layer
        assert np.all(np.asarray(d.expected) - np.asarray(d.actual) == 1)


@pytest.mark.integration
@pytest.mark.parametrize("sample_value", [1.0, 0.625, 0.5])
def test_cascaded_ttfs_two_core_hcm_sanafe_parity(sample_value):
    """core→core single-spike propagation: core0 fires **once** (single spike on
    the wire); core1 (``w1=0.5 < θ``) reconstructs the ramp from that single spike
    and must integrate it over **≥2** cycles before firing — also once.

    Answers "does SANA-FE send one spike per neuron and still decode correctly?":
    per-core output = single-spike *traffic* (exactly 1), and the decoded segment
    *value* matches HCM bit-for-bit."""
    if not have_sanafe():
        pytest.skip("SANA-FE not installed")
    import torch

    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

    ensure_backend_ready("sanafe")
    T = 8
    mapping = _build_two_core_cascade_mapping(w0=1.0, w1=0.5)
    behavior = _cascaded_behavior()

    flow = SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=mapping,
        simulation_length=T,
        preprocessor=torch.nn.Identity(),
        firing_mode=behavior.firing_mode,
        spike_mode=behavior.spike_generation_mode,
        thresholding_mode=behavior.thresholding_mode,
        spiking_mode=behavior.spiking_mode,
        ttfs_cycle_schedule="cascaded",
    ).eval()
    with torch.no_grad():
        _, ref = flow.forward_with_recording(
            torch.tensor([[sample_value]], dtype=torch.float32), sample_index=0,
        )
    ref_seg = ref.segments[0]

    runner = SanafeRunner(mapping=mapping, simulation_length=T, behavior=behavior)
    rec = runner.run(np.asarray([[sample_value]], dtype=np.float32), sample_index=0)
    act_seg = rec.segments[0]

    # Single-spike traffic: every fired neuron emits EXACTLY ONE spike, in BOTH
    # backends — the core of the hardware-faithfulness fix (no repeated spikes).
    hcm_core_out = [int(c.output_spike_count[0]) for c in ref_seg.cores]
    sanafe_core_out = [int(c.output_spike_count[0]) for c in act_seg.per_core]
    assert hcm_core_out == [1, 1], hcm_core_out
    assert sanafe_core_out == [1, 1], sanafe_core_out

    # Decoded segment VALUE (ramp from fire timing) matches bit-for-bit.
    assert list(ref_seg.seg_output_spike_count) == list(act_seg.seg_output_spike_count), (
        f"seg output differs: HCM={list(ref_seg.seg_output_spike_count)} "
        f"SANA-FE={list(act_seg.seg_output_spike_count)}"
    )
    # core1 fires strictly later than a same-input single-hop would (w1<θ forces
    # ≥2 ramp steps), so its decoded value is below the full window.
    assert 0 < int(act_seg.seg_output_spike_count[0]) < (T + 4)
