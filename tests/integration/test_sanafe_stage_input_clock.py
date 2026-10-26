"""[W33 finding] SANA-FE's synchronized-TTFS chip readout against the TTFS contract at atol 0 (built SANA-FE; slow).

Under synchronized TTFS SANA-FE chains the contract executor's values (``sanafe/runner/neural_stage.py``) and simulates
the chip beside them; the pipeline compares the chip's potential-trace readout with the contract at atol 1e-6
(``ttfs_recorder.TTFS_HARDWARE_ATOL``). The chip groups cores into synchronized windows by their ``ChipLatency``
latency (``net_synth/build.py``, ``latency_groups``), and the ALAP raise can give a producer the latency of its
consumer (skip: the stage-input reader is raised to latency 1, the latency of the core that reads it), so both land in
one window and the consumer reads its producer before it fires. The cause is dependency order (R6's class), not the
stage-input clock: the chip's soma weighs an input by its arrival time modulo the window, so moving a stage-input row
into its reader's window changes no readout (measured, the same 35 to 93 differing values with and without the
shift). The programs with such a pair are pinned strict-xfail; the chain, which has none, is the control."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import torch

from integration.parity_harness import ensure_sanafe
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.pass_timing import executed_core_latencies
from unit.mapping.residency.executed import vehicles as V


def _raised_input_readers(mapping) -> int:
    count = 0
    for stage in V.neural_passes(mapping):
        hcm = stage.hard_core_mapping
        for core, lat in zip(hcm.cores, executed_core_latencies(hcm)):
            count += lat > 0 and any(getattr(s, "is_input_", False) and not getattr(s, "is_off_", False)
                                     and not getattr(s, "is_always_on_", False) for s in core.axon_sources)
    return count


_SAME_WINDOW = pytest.mark.xfail(strict=True, reason="W33 finding: the ALAP raise puts a producer and its consumer in "
                                 "one synchronized window on SANA-FE's chip (dependency order, R6 class)")


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("kind", [pytest.param("skew3", marks=_SAME_WINDOW), pytest.param("skip", marks=_SAME_WINDOW),
                                  pytest.param("shift_probe", marks=_SAME_WINDOW), "chain"])
def test_the_synchronized_ttfs_readout_equals_the_contract(kind):
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract
    from mimarsinan.chip_simulation.ttfs.ttfs_recorder import compare_ttfs_records

    ensure_sanafe()
    ir = V.ir_of(V.chain_repr(seed=0) if kind == "chain" else V.branchy_repr(kind))
    identity = build_identity_hybrid_mapping(ir_graph=ir)
    assert (_raised_input_readers(identity) > 0) == (kind != "chain")
    behavior = NeuralBehaviorConfig(spiking_mode="ttfs_cycle_based", firing_mode="TTFS", thresholding_mode="<=",
                                    spike_generation_mode="TTFS")
    runner = SanafeRunner(mapping=identity, simulation_length=V.T, behavior=behavior,
                          ttfs_cycle_schedule="synchronized", pass_transfer="collapse")
    compared = 0
    for b in range(3):
        x = torch.rand(1, V.IN, generator=torch.Generator().manual_seed(b)).numpy().astype(np.float64)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = run_ttfs_hybrid_contract(identity, x, simulation_length=V.T, spiking_mode="ttfs_cycle_based",
                                           ttfs_cycle_schedule="synchronized").record
            rec = runner.run(x, sample_index=0)
        hardware = rec.to_ttfs_hardware_subset(spiking_mode="ttfs_cycle_based")
        assert compare_ttfs_records(ref, hardware, atol=0.0, rtol=0.0) == []
        compared += sum(len(seg.cores) for seg in hardware.segments.values())
    assert compared > 0
