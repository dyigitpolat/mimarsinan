"""[ODIN5, plan §7 row 18] The BARRIER bound dominates the measured drain.

The vendored tree carries no probe wire (plan §14, J3-7/J4-1): the runtime knows
a cycle's output is complete because it waited the sequencer program's own
DETERMINISTIC bound, computed at export time from the injected event count, the
512-cycle full-fan-out sweep, the 32-deep scheduler FIFO and the AER-out
handshake allowance. This gate makes that bound MEASURED rather than asserted,
on an adversarial case: a wide core at high fan-in whose neurons all fire, so
every output event stalls the controller on its handshake while the scheduler
still holds queued pre-synaptic rows.

An output event landing after `barrier_start + bound` is a failure -- it would
be counted in the next cycle's window on hardware, silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from integration.odin_rtl_harness import (
    export_of,
    hard_core,
    mapping_of,
    require_simulator,
    timed,
    traces_for,
)

from mimarsinan.chip_simulation.odin_rtl.cosim import run_cosim
from mimarsinan.chip_simulation.odin_rtl.stimulus import OP_AER
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.program import (
    NEURON_SWEEP_CYCLES,
    SCHEDULER_FIFO_DEPTH,
    drain_bound_cycles,
)
from mimarsinan.mapping.latency.chip import ChipLatency

pytestmark = [pytest.mark.slow, pytest.mark.integration]

#: Wide enough that a single cycle pushes far past the 32-deep spike FIFO and
#: every one of the 64 neurons fires on nearly every row event.
AXONS = 60
NEURONS = 64
S = 2


def _adversarial_mapping():
    """Maximum fan-in, unit threshold: every row event makes every neuron fire."""
    matrix = np.ones((AXONS + 1, NEURONS), dtype=np.float64)
    core = hard_core(
        matrix, threshold=1.0,
        sources=[SpikeSource(-2, i, is_input=True) for i in range(AXONS)]
        + [SpikeSource(-3, 0, is_always_on=True)],
    )
    mapping = mapping_of([core], [SpikeSource(0, i) for i in range(NEURONS)])
    ChipLatency(mapping).calculate()
    return mapping


@pytest.fixture(scope="module")
def adversarial():
    require_simulator()
    mapping = _adversarial_mapping()
    export = export_of(mapping)
    samples = traces_for(mapping, [[[1] * AXONS] * S], simulation_length=S)
    with timed("BARRIER adversarial cosim") as clock:
        result = run_cosim(
            export, [list(samples[0].per_cycle)],
            latencies=samples[0].trace.latencies)
    injected = sum(1 for op in result.plan.ops if op.code == OP_AER)
    print(f"[odin-rtl] barrier engine={result.build.engine} "
          f"sim={result.run.seconds:.1f}s cycles={result.capture.cycles} "
          f"bound={result.plan.barrier_cycles} aer_events={injected} "
          f"captured={len(result.capture.events)} wall={clock.seconds:.1f}s")
    return mapping, export, samples, result


class TestTheDeterministicBoundDominates:
    def test_no_output_event_lands_after_its_windows_deadline(self, adversarial):
        _mapping, _export, _samples, result = adversarial
        assert result.capture.drain_overruns() == ()

    def test_the_measured_drain_is_reported_against_the_bound(self, adversarial):
        _mapping, _export, _samples, result = adversarial
        measured = []
        for record in result.capture.barriers:
            last = result.capture.last_event_cycle(record.tag)
            start = result.capture.window_start_cycle(record.tag)
            if last is None or start is None:
                continue
            # From the instant the window opened -- pushes, sweeps, output
            # handshakes and all -- which is what the exported bound prices.
            measured.append(last - start)
        assert measured, "no window produced an output event to measure"
        print(f"[odin-rtl] measured drain per window: max={max(measured)} "
              f"bound={result.plan.barrier_cycles} "
              f"margin={result.plan.barrier_cycles - max(measured)}")
        assert max(measured) <= result.plan.barrier_cycles
        assert max(measured) > 0, "the window drained nothing measurable"

    def test_the_case_really_is_adversarial(self, adversarial):
        _mapping, _export, _samples, result = adversarial
        injected = sum(1 for op in result.plan.ops if op.code == OP_AER)
        # More pushes per cycle than the scheduler FIFO is deep, and an output
        # handshake per firing neuron.
        per_cycle = injected / max(1, result.plan.cycles_per_sample)
        assert per_cycle > SCHEDULER_FIFO_DEPTH, per_cycle
        assert len(result.capture.events) >= NEURONS

    def test_the_counts_are_still_the_folds_counts_under_the_pressure(
            self, adversarial):
        _mapping, _export, samples, result = adversarial
        for cycle, per_core in enumerate(samples[0].trace.outputs):
            expected = tuple(int(v) for v in per_core[0])
            assert result.cycle_counts(0, cycle, 0, NEURONS) == expected, cycle


class TestTheBoundIsTheExportersOwnFormula:
    def test_the_program_bound_is_the_closed_form_and_not_a_tuned_constant(
            self, adversarial):
        _mapping, export, _samples, result = adversarial
        barriers = [s["payload"]["cycles"] for s in export.program.stages
                    if s["kind"] == "BARRIER"]
        assert result.plan.barrier_cycles == max(barriers)
        assert max(barriers) >= SCHEDULER_FIFO_DEPTH * NEURON_SWEEP_CYCLES

    def test_the_bound_grows_with_the_injected_event_count(self):
        small = drain_bound_cycles(injected_events=1, emitted_spike_bound=0)
        large = drain_bound_cycles(injected_events=100, emitted_spike_bound=0)
        assert large > small
