"""[ODIN P7a] the Vitis RTL kernel: it elaborates, and its sequencer is the tb's.

Two gates, both local, both cheap compared with a HACC build hour:

  * ELABORATION — the packaged wrapper (`odin_fpga_kernel_top`, the module
    `package_xo` consumes) elaborates under iverilog. A syntax or width error
    in the kernel tree is otherwise discovered hours into a `v++` run on a
    cluster the owner has to log into with 2FA.

  * SEMANTICS — the SAME token program is executed twice against the same
    vendored core: once by the host-driven testbench (`hw/tb/tb_odin_core.v`,
    the P5 instrument that R11a certified) and once by the ON-FABRIC sequencer
    inside the kernel (`hw/fpga/kernel/odin_fpga_kernel.v`), and the per-neuron
    counts must be identical. That is what makes the board a TRANSPORT SWAP:
    the fabric executes the host's program with the host's semantics.

The witness is a per-EVENT one — a neuron that emits several spikes in a single
cycle — so a sequencer that collapsed multiplicities would fail here.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest

from integration.odin_rtl_harness import (
    ODIN_LAW,
    export_of,
    hard_core,
    mapping_of,
    require_simulator,
    timed,
)

from mimarsinan.chip_simulation.odin_fpga.kernel_sim import (
    KERNEL_TOP,
    elaborate_kernel_top,
    kernel_sources,
    run_kernel_program,
)
from mimarsinan.chip_simulation.odin_rtl.cosim import build_cosim_ops, run_cosim
from mimarsinan.chip_simulation.odin_rtl.reference import (
    per_slot_counts_by_cycle,
    simulate_cycles,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    ENGINE_INTERPRETED,
    SimulatorUnavailable,
    find_tool,
    unavailable_reason,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.latency.chip import ChipLatency

pytestmark = [pytest.mark.slow, pytest.mark.integration]

S = 2
AXONS = 4
NEURONS = 4
THETA = 3.0


def _micro_mapping():
    """One core: neuron 0 fires once per cycle, neuron 3 fires THREE times.

    Neuron 3 is fed by three separate axon slots, each on its own weight of 3
    against theta 3, so one input cycle delivers three ADJACENT crossings — the
    multiplicity only a per-event law produces.
    """
    weights = np.zeros((AXONS, NEURONS), dtype=np.float64)
    weights[0] = [5.0, 1.0, 0.0, 0.0]
    weights[1] = weights[2] = weights[3] = [0.0, 0.0, 0.0, 3.0]
    core = hard_core(
        weights, threshold=THETA,
        sources=[SpikeSource(-2, index, is_input=True) for index in range(AXONS)])
    mapping = mapping_of([core], [SpikeSource(0, i) for i in range(NEURONS)])
    ChipLatency(mapping).calculate()
    return mapping


def _fold(events) -> Dict[Tuple[int, int, int], int]:
    """``{(tag, core, neuron): count}`` — the tag-keyed per-cycle verdict.

    Cycles are NOT compared: the fabric sequencer's own clock schedule differs
    from the testbench driver's by construction (an FSM versus a behavioural
    task), and the deployment claim is about counts within a tagged window.
    """
    counts: Dict[Tuple[int, int, int], int] = {}
    for event in events:
        key = (int(event.tag), int(event.core), int(event.neuron))
        counts[key] = counts.get(key, 0) + 1
    return counts


@pytest.fixture(scope="module")
def micro_program():
    require_simulator()
    mapping = _micro_mapping()
    export = export_of(mapping)
    trace = simulate_cycles(
        mapping, soma_law=ODIN_LAW,
        input_counts=[[1] * AXONS for _ in range(S)], simulation_length=S,
        chip_latency=int(ChipLatency(mapping).calculate()))
    plan_inputs = [list(per_slot_counts_by_cycle(trace))]
    ops = build_cosim_ops(export, plan_inputs, latencies=trace.latencies).ops
    return mapping, export, trace, plan_inputs, ops


@pytest.fixture(scope="module")
def executions(micro_program):
    _mapping, export, trace, plan_inputs, ops = micro_program
    with timed("P7a host testbench") as host_clock:
        host = run_cosim(export, plan_inputs, latencies=trace.latencies)
    with timed("P7a fabric kernel") as kernel_clock:
        capture, build, run = run_kernel_program(ops, n_cores=1)
    print(
        f"[odin-kernel] engine={build.engine} tokens={host.token_count} "
        f"host_events={len(host.capture.events)} host_wall={host_clock.seconds:.1f}s "
        f"kernel_events={len(capture.events)} kernel_cycles={capture.cycles} "
        f"kernel_wall={kernel_clock.seconds:.1f}s")
    return host, capture


class TestTheWrapperElaborates:
    def test_the_packaged_kernel_elaborates_under_iverilog(self):
        if find_tool(ENGINE_INTERPRETED) is None:
            pytest.skip(unavailable_reason((ENGINE_INTERPRETED,)))
        with timed(f"P7a {KERNEL_TOP} elaboration"):
            elaborate_kernel_top(n_cores=1)

    def test_a_multi_core_kernel_elaborates_too(self):
        if find_tool(ENGINE_INTERPRETED) is None:
            pytest.skip(unavailable_reason((ENGINE_INTERPRETED,)))
        elaborate_kernel_top(n_cores=4)

    def test_the_kernel_tree_is_the_one_the_packaging_script_lists(self):
        names = {path.name for path in kernel_sources()}
        assert names == {
            "odin_spi_master.v", "odin_aer_bridge.v",
            "odin_fpga_kernel.v", "odin_fpga_kernel_top.v",
        }

    def test_the_simulator_is_present_or_the_gate_says_so(self):
        try:
            require_simulator()
        except SimulatorUnavailable as exc:  # pragma: no cover - loud skip path
            pytest.skip(str(exc))


class TestTheFabricSequencerIsTheTestbenchDriver:
    def test_the_counts_are_identical(self, executions):
        host, capture = executions
        assert _fold(capture.events) == _fold(host.capture.events)

    def test_the_kernel_captured_every_event_the_testbench_did(self, executions):
        host, capture = executions
        assert len(capture.events) == len(host.capture.events)
        assert capture.events, "the fabric kernel produced no output events"

    def test_the_witness_is_a_per_event_one(self, executions):
        host, capture = executions
        assert max(_fold(capture.events).values()) >= 3, (
            "no multiplicity in a single cycle: a sequencer that collapsed "
            "occurrences would reproduce this program and prove nothing")
        assert max(_fold(host.capture.events).values()) >= 3

    def test_the_fabric_matches_the_cycle_accurate_twin(self, executions, micro_program):
        _mapping, _export, trace, _plan, _ops = micro_program
        _host, capture = executions
        window = trace.window_counts()[0]
        got = [0] * NEURONS
        for (_tag, core, neuron), count in _fold(capture.events).items():
            assert core == 0
            got[neuron] += count
        assert tuple(got) == tuple(window[:NEURONS])
