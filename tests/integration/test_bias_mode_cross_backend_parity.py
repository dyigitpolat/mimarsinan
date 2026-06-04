"""Cross-backend spike-count parity for both bias modes, including deeper cores.

The offload-placed 2-layer model maps to a single neural segment with two cores
(latency 0 and 1), so the depth-1 core exercises the always-on-at-local-window
bias delivery that mode B (``param_encoded``) relies on. HCM and nevresim must
agree for both mode A (on-chip register) and mode B (always-on axon).
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.latency.chip import ChipLatency


def _ensure_nevresim():
    from integration.parity_harness import ensure_nevresim_ready

    ensure_nevresim_ready()


def _build_offload_flow(T, *, hw_bias, d_in=6, d_h=6, d_out=4):
    torch.manual_seed(0)
    p1 = Perceptron(d_h, d_in, normalization=nn.Identity(), base_activation_name="ReLU")
    p2 = Perceptron(d_out, d_h, normalization=nn.Identity(), base_activation_name="ReLU")
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(InputMapper((d_in,)), p1), p2))
    mark_encoding_layers(repr_, placement="offload")
    ir = IRMapping(q_max=127.0, firing_mode="TTFS", max_axons=64, max_neurons=64,
                   hardware_bias=hw_bias).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 64, "count": 8}])
    flow = SpikingHybridCoreFlow((d_in,), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="TTFS", spike_mode="TTFS", thresholding_mode="<=",
        spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded").double()
    return hybrid, flow, d_in


@pytest.mark.parametrize("hw_bias", [True, False])
def test_hcm_nevresim_cascaded_ttfs_parity_both_bias_modes(hw_bias):
    """HCM == nevresim output spike counts for the offload 2-core cascaded-TTFS
    segment, in both on-chip (mode A) and parameter-encoded (mode B) bias."""
    _ensure_nevresim()
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

    T = 8
    hybrid, flow, d_in = _build_offload_flow(T, hw_bias=hw_bias)
    seg = hybrid.stages[0].hard_core_mapping
    assert seg is not None and len(seg.cores) == 2, "expected a 2-core offload segment"

    torch.manual_seed(3)
    x = torch.rand(1, d_in, dtype=torch.float64)
    with torch.no_grad():
        hcm_out = flow(x.clone()).double().cpu().numpy()[0]  # output spike counts

    latency = ChipLatency(seg).calculate()
    flat = x.clone().cpu().numpy().astype(np.float64).reshape(-1)
    input_data = [(flat, np.array([0.0], dtype=np.float64))]
    with tempfile.TemporaryDirectory() as tmp:
        driver = NevresimDriver(
            d_in, seg, tmp, float,
            spike_generation_mode="TTFS", firing_mode="TTFS",
            thresholding_mode="<=", spiking_mode="ttfs_cycle_based", verbose=False,
        )
        raw = driver.predict_spiking_raw(input_data, T, latency)
    nev_out = np.rint(np.asarray(raw)[0]).astype(np.int64)

    np.testing.assert_array_equal(np.rint(hcm_out).astype(np.int64), nev_out)


@pytest.mark.slow
@pytest.mark.parametrize("hw_bias", [True, False])
def test_hcm_sanafe_cascaded_ttfs_parity_both_bias_modes(hw_bias):
    """SANA-FE == HCM for the offload 2-core cascaded-TTFS segment, both bias modes.

    Mode B (always-on axon) requires the bias spike to land at each consuming
    core's gated window start, so the depth-1 core's bias ramps in-window like HCM."""
    pytest.importorskip("sanafe")
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    from mimarsinan.chip_simulation.recording.compare import compare_records

    T = 8
    hybrid, flow, d_in = _build_offload_flow(T, hw_bias=hw_bias)
    torch.manual_seed(3)
    x = torch.rand(1, d_in, dtype=torch.float64)
    with torch.no_grad():
        _, ref = flow.forward_with_recording(x.clone().float(), sample_index=0)

    behavior = NeuralBehaviorConfig(
        spiking_mode="ttfs_cycle_based", firing_mode="TTFS",
        thresholding_mode="<=", spike_generation_mode="TTFS",
    )
    runner = SanafeRunner(
        mapping=hybrid, simulation_length=T, behavior=behavior,
        ttfs_cycle_schedule="cascaded",
    )
    rec = runner.run(x.clone().float().cpu().numpy(), sample_index=0)
    diffs = compare_records(ref, rec.to_hcm_subset())
    assert not diffs, f"SANA-FE↔HCM mode-{'A' if hw_bias else 'B'} diffs: {diffs}"
