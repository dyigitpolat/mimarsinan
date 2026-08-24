"""[ODIN P7a] the Vitis RTL kernel: it elaborates, moves its own bytes, and agrees.

Three gates, all local, all cheap compared with a HACC build hour:

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

  * DELIVERY — the WRAPPER (`odin_fpga_kernel_top`) is driven the way the XDMA
    shell drives it and no other way: the program is placed in a behavioural
    AXI4 memory model, split into the program and stimulus buffers a host hands
    the kernel, the six arguments go in over AXI4-Lite, ap_start is pulsed, and
    the capture is read back OUT OF THAT MEMORY. Its counts must equal the host
    testbench's. Nothing is preloaded into the fabric, so this is the gate that
    a hardwired-inert AXI master cannot pass.

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

from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    SHIPPED_CAPTURE_EVENTS,
    SHIPPED_CAPTURE_WORDS,
    STATUS_ERR_BIT,
    OdinFpgaCaptureTruncated,
    OdinFpgaKernelError,
    decode_capture,
    require_no_kernel_error,
)
from mimarsinan.chip_simulation.odin_fpga.kernel_sim import (
    KERNEL_TOP,
    elaborate_kernel_top,
    kernel_sources,
    run_kernel_program,
    run_kernel_program_over_axi,
)
from mimarsinan.chip_simulation.odin_rtl.cosim import build_cosim_ops, run_cosim
from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    OP_SHADOW,
    OP_TAG,
    OP_WAIT,
    Op,
)
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


@pytest.fixture(scope="module")
def over_axi(micro_program):
    """The same program, delivered through the wrapper's own DMA engine."""
    _mapping, _export, _trace, _plan_inputs, ops = micro_program
    with timed("P7a wrapper over AXI") as clock:
        capture, status, build, _run = run_kernel_program_over_axi(
            ops, n_cores=1)
    print(
        f"[odin-wrapper] engine={build.engine} events={len(capture.events)} "
        f"cycles={capture.cycles} err={int(status.err)} "
        f"seen={status.events_seen} cap_events={status.capture_capacity} "
        f"prog_words={status.program_words} stim_words={status.stimulus_words} "
        f"ram_words={status.ram_words} wall={clock.seconds:.1f}s")
    return capture, status


class TestTheWrapperMovesItsOwnBytes:
    """D1: the DMA engine, proven against a behavioural AXI4 memory model."""

    def test_the_counts_survive_the_round_trip_through_host_memory(
        self, over_axi, executions,
    ):
        host, _fabric = executions
        capture, _status = over_axi
        assert _fold(capture.events) == _fold(host.capture.events)

    def test_the_witness_is_still_a_per_event_one_after_the_dma(self, over_axi):
        capture, _status = over_axi
        assert max(_fold(capture.events).values()) >= 3, (
            "no multiplicity survived the DMA: a delivery that dropped or "
            "duplicated beats would show up exactly here")

    def test_the_program_really_arrived_in_two_buffers(self, over_axi):
        _capture, status = over_axi
        # The stimulus buffer is not a formality: the split lands inside the
        # real token stream, so a DMA that read only the program buffer would
        # execute a truncated program.
        assert status.program_words > 1 and status.stimulus_words > 1
        assert status.program_words + status.stimulus_words - 1 <= status.ram_words

    def test_the_run_was_clean_and_within_capacity(self, over_axi):
        capture, status = over_axi
        assert not status.err
        assert status.events_seen == len(capture.events)
        assert status.records_written == len(capture.events)
        assert status.events_seen < status.capture_capacity

    def test_the_capacity_registers_report_the_fabric_geometry(self, over_axi):
        """The SHIPPED capture depth, read off 0x54 and agreeing with the
        host-side copy of it — a fabric built at a depth the host does not
        believe would report a capacity nobody declared."""
        _capture, status = over_axi
        assert status.capture_capacity == SHIPPED_CAPTURE_EVENTS
        assert status.capture_capacity == (SHIPPED_CAPTURE_WORDS - 2) // 4
        assert status.program_capacity == status.ram_words


class TestTheKernelIsHonestWhenItCannotComply:
    """A1/A4: capacity and `err` reach the host instead of reading as silence."""

    def test_a_capture_over_capacity_stops_at_the_limit_and_reports_the_truth(
        self, micro_program,
    ):
        _mapping, _export, _trace, _plan_inputs, ops = micro_program
        # An 8-word capture RAM holds the two-word header and ONE record; the
        # program emits more than that.
        with timed("P7a wrapper capture overflow"):
            capture, status, _build, _run = run_kernel_program_over_axi(
                ops, n_cores=1, cap_words=8)
        assert status.capture_capacity == 1
        assert status.events_seen > status.capture_capacity, (
            "the overflow witness did not overflow: raise the program's event "
            "count or lower CAP_WORDS")
        assert status.records_written == status.capture_capacity
        assert len(capture.events) == status.capture_capacity
        assert status.truncated
        # ... and the numbers the FABRIC reported, handed to the HOST's own
        # decoder with the capacity the HOST read off 0x54, refuse the run.
        with pytest.raises(OdinFpgaCaptureTruncated):
            decode_capture(
                (status.events_seen, capture.cycles),
                capacity=status.capture_capacity)

    def test_an_unimplemented_opcode_raises_err_on_the_status_register(self):
        # SHADOW has no fabric implementation (config registers have no
        # readback path on silicon), so the sequencer must REFUSE it.
        with timed("P7a wrapper bad opcode"):
            _capture, status, _build, _run = run_kernel_program_over_axi(
                [Op(OP_SHADOW, (0, 0, 0))], n_cores=1)
        assert status.err
        assert status.events_seen == 0
        # The host reads that same status word and REFUSES by name rather than
        # decoding a capture the fabric never filled.
        with pytest.raises(OdinFpgaKernelError, match="REFUSED"):
            require_no_kernel_error(
                STATUS_ERR_BIT | status.events_seen, transport="xrt")

    def test_the_same_program_without_the_bad_opcode_does_not_raise_err(self):
        with timed("P7a wrapper clean control program"):
            _capture, status, _build, _run = run_kernel_program_over_axi(
                [Op(OP_TAG, (7,)), Op(OP_WAIT, (32,))], n_cores=1)
        assert not status.err


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
