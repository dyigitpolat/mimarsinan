"""[C2/WS-V] Cross-sim honesty proof for the nevresim final-membrane read port.

Compiles real nevresim binaries for a small LIF fixture and pins:
1. torch-hybrid membrane-decode logits == nevresim ``counts + m_T/theta``
   (fp tolerance) — the deployed membrane read matches the exact charge
   identity ``Q_T = theta*c_T + m_T``;
2. the spike-count output of the ``NEVRESIM_EXPORT_MEMBRANE`` build is
   bit-identical to the default build (the flag never touches the count
   currency);
3. a default build emits no membrane records (fail-loud on a membrane
   request against a non-export binary).
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from integration.parity_harness import ensure_nevresim_ready, have_cxx_compiler

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

import torch.nn as nn

pytestmark = pytest.mark.integration

# Sub-saturation drives (|w| <= theta): nevresim's established count semantics
# run output cores one overhang cycle (ChipLatency = deepest delay + 1), where
# a residual >= theta would refire; the membrane read is window-end gated and
# unaffected, but the fixture stays in the regime where counts are cross-sim
# bit-identical.
WEIGHTS = np.array([-0.4, 0.11, 0.25, 0.5, 0.73, 0.97], dtype=np.float64)
T = 8
SAMPLE_RATES = (1.0, 0.875, 0.5, 0.25)


def _constant_drive_fixture() -> HybridHardCoreMapping:
    """One latency-0 core, one input axon, ``len(WEIGHTS)`` output neurons."""
    n = len(WEIGHTS)
    core = HardCore(1, n, has_bias_capability=False)
    core.core_matrix = WEIGHTS.reshape(1, n).astype(np.float64)
    core.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core.available_axons = 0
    core.available_neurons = 0
    core.threshold = 1.0
    core.latency = 0

    segment = HardCoreMapping([])
    segment.cores = [core]
    segment.output_sources = np.asarray(
        [SpikeSource(0, i) for i in range(n)], dtype=object
    )
    stage = HybridStage(
        kind="neural",
        name="membrane_fixture",
        hard_core_mapping=segment,
        input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
        output_map=[SegmentIOSlice(node_id=0, offset=0, size=n)],
    )
    return HybridHardCoreMapping(
        stages=[stage],
        output_sources=np.asarray(
            [IRSource(node_id=0, index=i) for i in range(n)], dtype=object
        ),
    )


def _torch_flow(hybrid, *, membrane_readout: bool) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow(
        input_shape=(1,),
        hybrid_mapping=hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<=",
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
        membrane_readout=membrane_readout,
        membrane_readout_half_step=False,
    ).eval()


def _make_driver(segment, tmp, connectivity_mode="runtime"):
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

    return NevresimDriver(
        1,
        segment,
        tmp,
        float,
        spike_generation_mode="Uniform",
        firing_mode="Default",
        thresholding_mode="<=",
        spiking_mode="lif",
        connectivity_mode=connectivity_mode,
        verbose=False,
    )


def _input_loader():
    return [
        (np.asarray([rate], dtype=np.float64), np.zeros(1, dtype=np.float64))
        for rate in SAMPLE_RATES
    ]


@pytest.mark.parametrize("connectivity_mode", ["runtime", "compile_time"])
@pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
def test_membrane_export_cross_sim_agreement(connectivity_mode) -> None:
    ensure_nevresim_ready()
    hybrid = _constant_drive_fixture()
    segment = hybrid.stages[0].hard_core_mapping
    assert segment is not None
    latency = ChipLatency(segment).calculate()
    loader = _input_loader()

    with tempfile.TemporaryDirectory() as tmp:
        driver = _make_driver(segment, tmp, connectivity_mode)
        counts_default = driver.predict_spiking_raw(loader, T, latency, num_proc=2)
        counts_export, membranes = driver.predict_spiking_raw_with_membrane(
            loader, T, latency, num_proc=2,
        )

    # (2) count currency bit-identical export-build vs default build.
    np.testing.assert_array_equal(counts_export, counts_default)

    # (1) deployed nevresim membrane read == torch membrane-decode read.
    x = torch.tensor([[r] for r in SAMPLE_RATES], dtype=torch.float32)
    with torch.no_grad():
        torch_counts = _torch_flow(hybrid, membrane_readout=False)(x)
        torch_membrane_logits = _torch_flow(hybrid, membrane_readout=True)(x)

    np.testing.assert_array_equal(
        torch_counts.numpy().astype(np.int64),
        counts_default.astype(np.int64),
    )
    nevresim_membrane_logits = counts_export + membranes
    np.testing.assert_allclose(
        nevresim_membrane_logits,
        torch_membrane_logits.to(torch.float64).numpy(),
        atol=1e-4,
    )
    # The membrane term is genuinely engaged (sign-carrying, sub-integer).
    assert np.abs(membranes).max() > 0.0


@pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
def test_default_build_yields_no_membrane_records() -> None:
    """(3) A default binary run with a membrane request fails loud instead of
    silently returning empty membranes."""
    from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw

    ensure_nevresim_ready()
    hybrid = _constant_drive_fixture()
    segment = hybrid.stages[0].hard_core_mapping
    assert segment is not None
    latency = ChipLatency(segment).calculate()
    loader = _input_loader()[:1]

    with tempfile.TemporaryDirectory() as tmp:
        driver = _make_driver(segment, tmp)
        binary = driver.emit_main_and_compile(1, T, latency)
        with pytest.raises(RuntimeError, match="membrane"):
            run_binary_raw(
                binary_path=binary,
                work_dir=tmp,
                input_loader=loader,
                output_size=driver.chip.output_size,
                simulation_length=T,
                input_size=driver.chip.input_size,
                spike_generation_mode="Uniform",
                max_input_count=1,
                export_membrane=True,
            )
