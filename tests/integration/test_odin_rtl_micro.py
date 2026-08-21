"""[ODIN5, plan §7 rows 9/15] Single-core micro-witnesses: the RTL IS the fold.

Four physical witnesses, all delivered as NEURON-SPIKE events (the only
stimulation the deployment contract uses — a virtual event would carry its
weight in the AER word instead of reading it out of the programmed crossbar,
and would prove nothing about the exported image):

  (i)   one row event fires exactly the neurons whose programmed weight crosses
        theta, and no others;
  (ii)  a multiplicity-3 slot delivered as three ADJACENT row events produces
        the fold's counts -- several spikes from one neuron in one cycle, which
        no per-cycle law can produce -- and a sign-asymmetric neuron on the same
        core fires only BECAUSE the occurrences stayed adjacent;
  (iii) the saturation rails: theta at the 8-bit ceiling with a charge that
        crosses 255 (upstream's `state_core <= state_exc` overflow test clamps
        to 8'hFF, which a plain 8-bit wrap would turn into a MISSED spike), and
        an inhibitory event below zero flooring at 0 rather than wrapping;
  (iv)  an inhibitory row pair of one logical slot, exercising the
        per-(slot, neuron) disjointness the row-pair lemma rests on.

Every expected number comes from the cycle-accurate twin around
``lif_serial_fold`` -- the same kernel both torch executors call -- so a
difference here is a difference between the fold and the silicon, not between
two hand-written tables.
"""

from __future__ import annotations

import numpy as np
import pytest

from integration.odin_rtl_harness import (
    MEMBRANE_CEILING,
    compare_cycle_counts,
    export_of,
    hard_core,
    mapping_of,
    require_simulator,
    timed,
    traces_for,
)

from mimarsinan.chip_simulation.odin_rtl.cosim import run_cosim
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.latency.chip import ChipLatency

pytestmark = [pytest.mark.slow, pytest.mark.integration]

S = 5
INPUT_SIZE = 14
RAIL_SLOTS = 9
RAIL_WEIGHT = 7

WITNESS_CORE = 0
MULTIPLICITY_CORE = 1
RAIL_CORE = 2


def _witness_mapping():
    """Three cores, one per threshold the witnesses need.

    Core 0 (theta 4) carries the single-row witness (neurons 0-1), the
    inhibitory row-pair witness (neurons 2-3) and the multiplicity PRODUCER
    (neuron 4: three unit-input slots, each supra-threshold, so one input spike
    leaves the core as three). Core 1 (theta 5) consumes that count. Core 2
    (theta 255) is the saturation-rail witness.
    """
    witness = np.zeros((7, 5), dtype=np.float64)
    witness[0] = [5.0, 1.0, 0.0, 0.0, 0.0]
    witness[2] = [0.0, 0.0, 3.0, -3.0, 0.0]
    witness[3] = [0.0, 0.0, -2.0, 5.0, 0.0]
    witness[4] = witness[5] = witness[6] = [0.0, 0.0, 0.0, 0.0, 5.0]
    core0 = hard_core(
        witness, threshold=4.0,
        sources=[SpikeSource(-2, i, is_input=True) for i in (0, 1, 2, 3, 4, 4, 4)],
    )

    consumer = np.array([[5.0, 0.0], [0.0, 3.0], [0.0, -3.0]], dtype=np.float64)
    core1 = hard_core(
        consumer, threshold=5.0,
        sources=[SpikeSource(WITNESS_CORE, 4)] * 3,
    )

    rail = np.full((RAIL_SLOTS, 1), float(RAIL_WEIGHT))
    core2 = hard_core(
        rail, threshold=float(MEMBRANE_CEILING),
        sources=[SpikeSource(-2, 5 + i, is_input=True) for i in range(RAIL_SLOTS)],
    )

    mapping = mapping_of(
        [core0, core1, core2],
        [SpikeSource(MULTIPLICITY_CORE, 0), SpikeSource(RAIL_CORE, 0)],
    )
    ChipLatency(mapping).calculate()
    return mapping


def _raster():
    """One entry spike per cycle on every line except input 1 (the silent one)."""
    row = [1] * INPUT_SIZE
    row[1] = 0
    return [list(row) for _ in range(S)]


@pytest.fixture(scope="module")
def witnesses():
    require_simulator()
    mapping = _witness_mapping()
    export = export_of(mapping)
    samples = traces_for(mapping, [_raster()], simulation_length=S)
    with timed("micro-witness cosim") as clock:
        result = run_cosim(
            export, [list(samples[0].per_cycle)],
            latencies=samples[0].trace.latencies)
    print(f"[odin-rtl] engine={result.build.engine} "
          f"build={result.build.build_seconds:.1f}s cached={result.build.cached} "
          f"sim={result.run.seconds:.1f}s cycles={result.capture.cycles} "
          f"tokens={result.token_count} wall={clock.seconds:.1f}s")
    return mapping, export, samples, result


class TestTheRtlReproducesTheFoldOnEveryWitness:
    def test_every_core_every_cycle_every_neuron_matches_at_zero_difference(
            self, witnesses):
        _mapping, _export, samples, result = witnesses
        assert compare_cycle_counts(result, samples) == []

    def test_a_single_row_event_fires_only_the_neuron_whose_weight_crosses_theta(
            self, witnesses):
        _mapping, _export, _samples, result = witnesses
        # Row 0 carries w=5 to neuron 0 (fires every cycle) and w=1 to neuron 1
        # (which needs four cycles of accumulation before it crosses theta=4).
        first = [result.cycle_counts(0, cycle, WITNESS_CORE, 5)
                 for cycle in range(S)]
        assert [row[0] for row in first] == [1] * S
        assert [row[1] for row in first] == [0, 0, 0, 1, 0]

    def test_a_multiplicity_three_slot_produces_three_spikes_in_one_cycle(
            self, witnesses):
        _mapping, _export, samples, result = witnesses
        producer = [result.cycle_counts(0, cycle, WITNESS_CORE, 5)[4]
                    for cycle in range(S)]
        assert producer == [3] * S, producer
        consumer = [result.cycle_counts(0, cycle, MULTIPLICITY_CORE, 2)[0]
                    for cycle in range(1, S)]
        assert consumer == [3] * (S - 1), consumer
        assert max(consumer) >= 2

    def test_the_sign_asymmetric_neuron_fires_only_under_adjacency(self, witnesses):
        _mapping, _export, _samples, result = witnesses
        # w = (+3, -3) on two slots that both carry the same count 3 and
        # theta = 5: adjacent gives one crossing, round-robin gives none.
        witness = [result.cycle_counts(0, cycle, MULTIPLICITY_CORE, 2)[1]
                   for cycle in range(1, S)]
        assert witness == [1] * (S - 1), witness

    def test_the_ceiling_rail_fires_where_an_eight_bit_wrap_would_miss(self, witnesses):
        _mapping, _export, _samples, result = witnesses
        rail = [result.cycle_counts(0, cycle, RAIL_CORE, 1)[0] for cycle in range(S)]
        # 9 slots x w=7 = 63 per cycle: 252 after four cycles, and the 253rd
        # charge overflows the register. Clamped to 255 it crosses theta=255;
        # wrapped to 3 it would not.
        assert rail == [0, 0, 0, 0, 1], rail

    def test_the_inhibitory_row_pair_floors_at_zero_instead_of_wrapping(
            self, witnesses):
        _mapping, _export, _samples, result = witnesses
        # Neuron 3 takes -3 on slot 2's inhibitory row while its membrane is 0,
        # then +5 on slot 3's excitatory row. A wrapping register would sit at
        # 253 and fire on the inhibitory event itself.
        neuron3 = [result.cycle_counts(0, cycle, WITNESS_CORE, 5)[3]
                   for cycle in range(S)]
        assert neuron3 == [1] * S, neuron3
        neuron2 = [result.cycle_counts(0, cycle, WITNESS_CORE, 5)[2]
                   for cycle in range(S)]
        assert neuron2 == [0, 1, 0, 1, 0], neuron2


class TestTheWitnessesAreNonDegenerate:
    def test_the_fixture_exercises_both_members_of_a_row_pair(self, witnesses):
        _mapping, export, _samples, _result = witnesses
        stage = next(s for s in export.program.stages
                     if s["kind"] == "INJECT" and s["payload"]["core_index"] == 0)
        runtime_rows = stage["payload"]["runtime_rows"]
        paired = [slot for slot, slot_rows in runtime_rows if len(slot_rows) == 2]
        assert paired, runtime_rows

    def test_the_saturation_witness_sits_at_the_membrane_ceiling(self, witnesses):
        mapping, export, _samples, _result = witnesses
        assert mapping.cores[RAIL_CORE].threshold == MEMBRANE_CEILING
        assert export.manifest["feasibility"]["theta_per_core"][str(RAIL_CORE)] \
            == MEMBRANE_CEILING

    def test_the_counts_on_the_wire_exceed_one_so_the_law_is_visibly_per_event(
            self, witnesses):
        _mapping, _export, samples, _result = witnesses
        assert max(
            max(counts) for per_core in samples[0].trace.outputs for counts in per_core
        ) >= 3
