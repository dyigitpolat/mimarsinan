"""The code generation of a timing program refuses, by name, a decision that 64 bits cannot hold.

The integer decision forms the membrane numerator m over 2S (below 2^53: the reference's own reach, read per sample
on both sides) and the doubled threshold 2 theta in 64-bit integers, as the Python executor does
(``ttfs_quantized_staircase_lattice_np``). An integral threshold from 2^62 on has no 64-bit doubled form, so a timing
program carrying one is refused where its compiled program is generated, on nevresim and on SANA-FE alike.
"""

from __future__ import annotations

import tempfile

from spiking_points import point_of_mode

import numpy as np
import pytest

from .test_exact_output import _one_core_segment

pytestmark = [pytest.mark.slow]

HUGE_THETA = float(2 ** 62)


def _refusal_type():
    try:
        from mimarsinan.chip_simulation.timing_decision import TimingDecisionRangeError
    except ImportError:
        return None
    return TimingDecisionRangeError


@pytest.mark.timeout(600)
def test_nevresim_generation_refuses_a_threshold_without_a_64_bit_double():
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.common.env import nevresim_root

    NevresimDriver.nevresim_path = NevresimDriver.nevresim_path or nevresim_root()
    refused = None
    with tempfile.TemporaryDirectory() as tmp:
        try:
            NevresimDriver(
                1, _one_core_segment(3.0, HUGE_THETA), tmp, int,
                spike_generation_mode="TTFS", thresholding_mode="<=", firing_mode="TTFS",
                point=point_of_mode("ttfs_quantized"), threshold_type=float, verbose=False,
            )
        except Exception as exc:  # the named refusal, read by its type below
            refused = exc
    assert refused is not None, "the program's code was generated: no refusal"
    named = _refusal_type()
    assert named is not None and isinstance(refused, named), repr(refused)
    assert "2^62" in str(refused)


@pytest.mark.timeout(600)
def test_nevresim_generation_takes_the_largest_threshold_it_can_hold():
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.common.env import nevresim_root

    NevresimDriver.nevresim_path = NevresimDriver.nevresim_path or nevresim_root()
    with tempfile.TemporaryDirectory() as tmp:
        NevresimDriver(
            1, _one_core_segment(3.0, float(2 ** 61)), tmp, int,
            spike_generation_mode="TTFS", thresholding_mode="<=", firing_mode="TTFS",
            point=point_of_mode("ttfs_quantized"), threshold_type=float, verbose=False,
        )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("schedule", ["quantized", "synchronized"])
def test_sanafe_synthesis_refuses_a_threshold_without_a_64_bit_double(schedule):
    pytest.importorskip("sanafe")
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    from mimarsinan.mapping.ir import IRSource
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        HybridHardCoreMapping,
        HybridStage,
        SegmentIOSlice,
    )

    mode = "ttfs_quantized" if schedule == "quantized" else "ttfs_cycle_based"
    segment = _one_core_segment(3.0, HUGE_THETA)
    hybrid = HybridHardCoreMapping(
        stages=[HybridStage(kind="neural", name="one_core", hard_core_mapping=segment,
                            input_map=[SegmentIOSlice(node_id=-2, offset=0, size=1)],
                            output_map=[SegmentIOSlice(node_id=0, offset=0, size=1)])],
        output_sources=np.asarray([IRSource(node_id=0, index=0)], dtype=object),
    )
    behavior = NeuralBehaviorConfig(spiking_mode=mode, firing_mode="TTFS", thresholding_mode="<=",
                                    spike_generation_mode="TTFS")
    runner = SanafeRunner(mapping=hybrid, simulation_length=8, behavior=behavior,
                          ttfs_cycle_schedule="synchronized" if schedule == "synchronized" else "cascaded")
    refused = None
    try:
        runner.run(np.asarray([[1.0]], dtype=np.float32), sample_index=0)
    except Exception as exc:  # the named refusal, read by its type below
        refused = exc
    assert refused is not None, "the program ran: no refusal"
    named = _refusal_type()
    assert named is not None and isinstance(refused, named), repr(refused)
