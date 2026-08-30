"""[ODIN C2] the WIDE chip configuration IN THE WRAPPER: the fabric is the variant.

The stock kernel instantiates the vendored ODIN core and programs it over SPI.
This gate proves the SAME Vitis wrapper -- byte-identical `odin_fpga_kernel_top`,
the same AXI4-Lite register map, the same six kernel arguments, the same capture
layout -- driving the GENERATED 1024x256 core through its direct configuration
port, and reproducing the host-driven cosimulation's counts exactly.

WHAT SELECTS THE FABRIC IS THE SOURCE SET, not a parameter. The variant kernel
declares the same module name (`odin_fpga_kernel`) with the same port list, so
`odin_fpga_kernel_top.v` binds either without an edit, and the stock kernel file
-- the RTL a routed bitstream was built from, and the file the chip cache's RTL
digest covers -- is never touched. `chip_configs.ChipConfig.rtl_sources` is the
one place that says which files a fabric is made of, and `scripts/hacc/chips.sh`
is its shell copy.

Four gates, mirroring the stock kernel's:

  * ELABORATION -- the packaged wrapper elaborates over the wide source set.
  * SEMANTICS -- the same token program run twice against the same generated
    core, once by the host-driven testbench (`hw/tb/tb_odin_gen_core.v`, the
    instrument the geometry gate certified) and once by the ON-FABRIC sequencer,
    at identical per-neuron counts. The programming form is the difference the
    fabric had to learn: OP_PROG, four arguments, one clock on the config port.
  * DELIVERY -- the WRAPPER's own AXI4 DMA engine moves the whole program (two
    buffers, a 65,536-word synapse image among them) into the elastic FIFO while
    the sequencer executes it, and the capture comes back out of host memory.
  * STALL INVARIANCE -- five seeded starvations of the AXI read data channel
    against the no-stall baseline, byte-identical events at byte-identical
    ENABLED-cycle timestamps.

Plus the refusal the wide fabric owns: it has no SPI slave, so an SPI opcode is
`err` rather than a program nobody assembled.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest

from integration.odin_gen_harness import (
    hard_core,
    images_for,
    mapping_of,
    report,
    require_simulator,
    timed,
    traces_for,
)

from mimarsinan.chip_simulation.odin_fpga.chip_configs import (
    WIDE_CHIP,
    chip_config_named,
)
from mimarsinan.chip_simulation.odin_fpga.kernel_sim import (
    KERNEL_TOP,
    elaborate_kernel_top,
    run_kernel_program,
    run_kernel_program_over_axi,
)
from mimarsinan.chip_simulation.odin_rtl.gen_cosim import (
    build_variant_ops,
    run_variant_cosim,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import OP_SPI_W, OP_TAG, OP_WAIT, Op
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    ENGINE_INTERPRETED,
    find_tool,
    unavailable_reason,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin_gen import generate_core

pytestmark = [pytest.mark.slow, pytest.mark.integration]

CHIP = chip_config_named(WIDE_CHIP)
SPEC = CHIP.core_spec
AXONS = SPEC.max_axons
NEURONS = SPEC.max_neurons
THETA = 300.0
S = 1
CYCLES = 2

#: 784 input lines plus the always-on bias row: what 1024 rows are FOR.
MNIST_INPUT_LINES = 784

#: The seeds of the stall-invariance gate; 0 is the no-stall baseline.
STALL_SEEDS = (0, 1, 4919, 31337)


def _mapping():
    """One wide core: a multiplicity witness, the top row, the last neuron."""
    matrix = np.zeros((AXONS, NEURONS), dtype=np.float64)
    for row in range(6):
        matrix[row][0] = 100.0            # 600 against theta 300: TWO crossings
    for row in (900, 901, 902):
        matrix[row][1] = 100.0            # driven from above the MNIST width
    matrix[AXONS - 1][1] = -128.0         # the extreme negative two's-complement cell
    for row in (MNIST_INPUT_LINES - 1, MNIST_INPUT_LINES, MNIST_INPUT_LINES + 1):
        matrix[row][NEURONS - 1] = 100.0  # the LAST neuron address
    core = hard_core(
        matrix, threshold=THETA,
        sources=[SpikeSource(-2, index, is_input=True) for index in range(AXONS)],
    )
    return mapping_of([core], [SpikeSource(0, 0)])


def _raster():
    row = [0] * AXONS
    driven = (list(range(6)) + [900, 901, 902]
              + [MNIST_INPUT_LINES - 1, MNIST_INPUT_LINES, MNIST_INPUT_LINES + 1]
              + [AXONS - 1])
    for index in driven:
        row[index] = 1
    return [list(row) for _ in range(CYCLES)]


def _fold(events) -> Dict[Tuple[int, int, int], int]:
    """``{(tag, core, neuron): count}`` -- the tag-keyed per-cycle verdict.

    Cycles are NOT compared: the fabric sequencer's clock schedule differs from
    the testbench driver's by construction, and the claim is about counts inside
    a tagged window.
    """
    counts: Dict[Tuple[int, int, int], int] = {}
    for event in events:
        key = (int(event.tag), int(event.core), int(event.neuron))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _timed_events(capture):
    """The capture as a comparable image: every field, timestamps INCLUDED."""
    return tuple(
        (int(e.tag), int(e.core), int(e.neuron), int(e.cycle))
        for e in capture.events)


@pytest.fixture(scope="module")
def wide_program():
    """The token program, and the RTL source set the wide chip config declares."""
    require_simulator()
    mapping = _mapping()
    generated = generate_core(SPEC)
    images = images_for(mapping, SPEC)
    samples = traces_for(
        mapping, [_raster()], soma_law=SPEC.soma_law, simulation_length=CYCLES)
    per_cycle = [list(sample.per_cycle) for sample in samples]
    plan = build_variant_ops(
        images, per_cycle, spec=SPEC, latencies=samples[0].trace.latencies)
    sources = list(CHIP.rtl_sources())
    return mapping, generated, images, samples, per_cycle, plan, sources


@pytest.fixture(scope="module")
def host_reference(wide_program):
    """The host-driven variant testbench: the instrument the geometry gate proved."""
    _map, generated, images, samples, per_cycle, _plan, _sources = wide_program
    with timed("C2 host variant testbench"):
        result = run_variant_cosim(
            generated, images, per_cycle,
            latencies=samples[0].trace.latencies)
    report("wide-host", result)
    return result


@pytest.fixture(scope="module")
def fabric_run(wide_program):
    """The SAME program, executed by the on-fabric sequencer of the wide kernel."""
    _map, _gen, _images, _samples, _per_cycle, plan, sources = wide_program
    with timed("C2 wide fabric kernel") as clock:
        capture, build, _run = run_kernel_program(
            list(plan.ops), n_cores=1, sources=sources)
    print(f"[odin-wide] engine={build.engine} events={len(capture.events)} "
          f"cycles={capture.cycles} wall={clock.seconds:.1f}s")
    return capture


@pytest.fixture(scope="module")
def wrapper_runs(wide_program):
    """The same program through the WRAPPER's DMA, at every stall seed."""
    _map, _gen, _images, _samples, _per_cycle, plan, sources = wide_program
    runs = {}
    for seed in STALL_SEEDS:
        with timed(f"C2 wide wrapper stall seed {seed}") as clock:
            capture, status, _build, _run = run_kernel_program_over_axi(
                list(plan.ops), n_cores=1, stall_seed=seed, sources=sources)
        runs[seed] = (capture, status)
        print(f"[odin-wide-stall] seed={seed} events={len(capture.events)} "
              f"enabled_cycles={capture.cycles} err={int(status.err)} "
              f"gaps={status.stall_gaps} starved_cycles={status.stall_cycles} "
              f"prog_words={status.program_words} "
              f"stim_words={status.stimulus_words} wall={clock.seconds:.1f}s")
    return runs


class TestTheWideFabricElaborates:
    def test_the_packaged_wrapper_elaborates_over_the_wide_source_set(
            self, wide_program):
        if find_tool(ENGINE_INTERPRETED) is None:
            pytest.skip(unavailable_reason((ENGINE_INTERPRETED,)))
        *_rest, sources = wide_program
        with timed(f"C2 {KERNEL_TOP} elaboration (wide)"):
            elaborate_kernel_top(n_cores=1, sources=sources)

    def test_a_multi_core_wide_kernel_elaborates_too(self, wide_program):
        if find_tool(ENGINE_INTERPRETED) is None:
            pytest.skip(unavailable_reason((ENGINE_INTERPRETED,)))
        *_rest, sources = wide_program
        elaborate_kernel_top(n_cores=4, sources=sources)

    def test_the_wide_source_set_carries_no_stock_kernel_and_no_vendored_core(
            self, wide_program):
        """The whole reason the stock RTL digest is unmoved: this fabric does not
        compile the stock kernel body or the vendored tree at all."""
        *_rest, sources = wide_program
        names = [path.name for path in sources]
        parents = {path.parent.name for path in sources}
        assert "ODIN.v" not in names
        assert "odin_spi_master.v" not in names
        assert "vendor" not in parents and "src" not in parents
        assert names.count("odin_fpga_kernel.v") == 1
        assert "odin_fpga_kernel_top.v" in names
        # ... and the one that IS named `odin_fpga_kernel.v` is the GENERATED
        # one, not the committed stock body.
        kernel = next(p for p in sources if p.name == "odin_fpga_kernel.v")
        assert kernel.parent.name == WIDE_CHIP


class TestTheWideFabricSequencerIsTheTestbenchDriver:
    def test_the_counts_are_identical(self, fabric_run, host_reference):
        assert _fold(fabric_run.events) == _fold(host_reference.capture.events)

    def test_the_host_reference_is_not_vacuous(self, host_reference):
        assert host_reference.capture.events
        assert host_reference.capture.spec_failures == 0
        assert host_reference.capture.rail_failures == 0

    def test_the_witness_is_a_per_event_one(self, fabric_run):
        assert max(_fold(fabric_run.events).values()) >= 2, (
            "no multiplicity in a single cycle: a sequencer that collapsed "
            "occurrences would reproduce this program and prove nothing")

    def test_the_fabric_drove_the_widened_row_and_neuron_addresses(
            self, fabric_run):
        neurons = {neuron for (_tag, _core, neuron) in _fold(fabric_run.events)}
        assert NEURONS - 1 in neurons, (
            "the last neuron address never fired, so its rows -- the ones above "
            "the MNIST width -- were never swept")
        assert 1 in neurons


class TestTheWideWrapperMovesItsOwnBytes:
    def test_the_counts_survive_the_round_trip_through_host_memory(
            self, wrapper_runs, host_reference):
        capture, _status = wrapper_runs[STALL_SEEDS[0]]
        assert _fold(capture.events) == _fold(host_reference.capture.events)

    def test_the_program_really_arrived_in_two_buffers(self, wrapper_runs):
        _capture, status = wrapper_runs[STALL_SEEDS[0]]
        assert status.program_words > 1 and status.stimulus_words > 1
        assert status.stream_words == (
            status.program_words + status.stimulus_words - 1)

    def test_the_whole_synapse_image_crossed_the_bus(self, wrapper_runs):
        """A 1024x256 core at 8 bits is 65,536 programming words; the DMA moved
        every one of them, five tokens each, while the sequencer ran."""
        _capture, status = wrapper_runs[STALL_SEEDS[0]]
        assert status.stream_words > 5 * SPEC.synapse_depth

    def test_the_run_was_clean_and_within_capacity(self, wrapper_runs):
        capture, status = wrapper_runs[STALL_SEEDS[0]]
        assert not status.err
        assert status.events_seen == len(capture.events)
        assert status.records_written == len(capture.events)
        assert status.events_seen < status.capture_capacity

    def test_the_capacity_the_fabric_reports_is_the_chip_config_declaration(
            self, wrapper_runs):
        _capture, status = wrapper_runs[STALL_SEEDS[0]]
        assert status.capture_capacity == CHIP.kernel_capacity().capture_events


class TestTheWideStreamMayArriveHoweverItLikes:
    def test_every_seed_saw_a_genuinely_different_arrival_pattern(
            self, wrapper_runs):
        baseline = wrapper_runs[STALL_SEEDS[0]][1]
        assert (baseline.stall_gaps, baseline.stall_cycles) == (0, 0)
        injected = {
            seed: (wrapper_runs[seed][1].stall_gaps,
                   wrapper_runs[seed][1].stall_cycles)
            for seed in STALL_SEEDS[1:]
        }
        assert all(gaps > 0 and cycles > 0 for gaps, cycles in injected.values()), (
            f"a seed injected no starvation at all ({injected}): this gate "
            f"would be vacuous")
        assert len(set(injected.values())) == len(injected)

    def test_the_events_are_identical_under_every_starvation(self, wrapper_runs):
        images = {seed: _timed_events(run[0]) for seed, run in wrapper_runs.items()}
        baseline = images[STALL_SEEDS[0]]
        assert baseline, "the stall-invariance witness produced no events"
        for seed in STALL_SEEDS[1:]:
            assert images[seed] == baseline, (
                f"seed {seed} produced a different capture than the no-stall "
                f"baseline: the arrival pattern of the op stream changed the "
                f"program on the wide fabric, which is exactly what core-enable "
                f"gating exists to make impossible")

    def test_the_enabled_cycle_clock_is_identical_too(self, wrapper_runs):
        cycles = {seed: run[0].cycles for seed, run in wrapper_runs.items()}
        assert len(set(cycles.values())) == 1, (
            f"the enabled-cycle count moved with the arrival pattern: {cycles}")

    def test_no_seed_dropped_a_word_or_raised_err(self, wrapper_runs):
        for seed, (capture, status) in wrapper_runs.items():
            assert not status.err, f"seed {seed} raised err"
            assert status.events_seen == len(capture.events)
            assert status.stall_seed == seed


class TestTheWideFabricRefusesWhatItCannotDo:
    def test_an_spi_opcode_raises_err_because_there_is_no_spi_slave(
            self, wide_program):
        """The mirror of the stock kernel's OP_PROG refusal: the generated core
        is programmed through a synchronous port, so an SPI transaction has
        nothing to drive and the sequencer must REFUSE rather than skip it."""
        *_rest, sources = wide_program
        with timed("C2 wide wrapper SPI opcode"):
            _capture, status, _build, _run = run_kernel_program_over_axi(
                [Op(OP_SPI_W, (0, 0, 0))], n_cores=1, sources=sources)
        assert status.err
        assert status.events_seen == 0

    def test_the_same_control_program_without_it_does_not_raise_err(
            self, wide_program):
        *_rest, sources = wide_program
        with timed("C2 wide wrapper clean control program"):
            _capture, status, _build, _run = run_kernel_program_over_axi(
                [Op(OP_TAG, (7,)), Op(OP_WAIT, (32,))], n_cores=1, sources=sources)
        assert not status.err
