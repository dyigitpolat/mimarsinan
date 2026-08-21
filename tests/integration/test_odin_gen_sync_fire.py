"""[ODIN6, plan §7 row 20] The sync-fire flagship: a signed 16-bit membrane.

`firing_granularity='per_cycle'` on a two's-complement register — the value the
plan deferred (§11) until a target needed it, un-deferred here because a
per-cycle window whose NET charge goes negative cannot be held by a register
that floors at zero, and the sync-fire core's contract is that it holds exactly
the number the unbounded accumulator holds.

The fixture is built around that case and one more: neuron 0 of the producer
takes +3 and -7 in one cycle, sits at -4, and only crosses theta two cycles
later. An unsigned register floors at 0 on the first cycle and therefore fires
EARLY — the gate pins that difference so the comparison cannot be vacuous.
Neuron 1 takes TWO events that each alone reach theta in one cycle, which is
the only stimulus under which the per-cycle law and the event-serial law
disagree: the per-cycle compare emits one spike where a serial fold emits two.

Four things are compared at zero difference: the generated RTL, the deployed
torch per-cycle kernels, the nevresim `WholeVectorSaturatingSigned` policy, and
the UNBOUNDED law the whole variant exists to reproduce. The static
no-saturation bound is computed by the real gate before any of them runs, and
the RTL's sticky rail flag is asserted clear afterwards.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from integration.odin_gen_harness import (
    compare_cycle_counts,
    hard_core,
    images_for,
    mapping_of,
    report,
    require_simulator,
    rtl_cycle_multiplicity,
    spec_for,
    suprathreshold_multiplicity,
    sync_fire_law,
    thetas_of,
    timed,
    traces_for,
    unbounded_law,
)
from integration.parity_harness import ensure_nevresim_ready, have_cxx_compiler

from mimarsinan.chip_simulation.odin_rtl.gen_cosim import run_variant_cosim
from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin_gen import generate_core
from mimarsinan.mapping.export.odin_gen.feasibility import require_no_saturation

pytestmark = [pytest.mark.slow, pytest.mark.integration]

AXONS = 256
NEURONS = 256
MEMBRANE_BITS = 16
S = 5
THETA = 5.0
PRODUCER = 0
CONSUMER = 1


def _producer():
    """Neuron 0 is the net-negative witness; neuron 1 the law witness.

    Rows 2 and 3 both reach theta on their own, so neuron 1 is the neuron that
    receives two suprathreshold events in one cycle; neuron 255 exercises the
    top address.
    """
    matrix = np.zeros((AXONS, NEURONS), dtype=np.float64)
    matrix[0][0] = 3.0
    matrix[1][0] = -7.0
    matrix[2][1] = 5.0
    matrix[3][1] = 5.0
    matrix[AXONS - 1][NEURONS - 1] = 5.0
    return matrix


def _consumer():
    matrix = np.zeros((AXONS, NEURONS), dtype=np.float64)
    matrix[0][0] = 5.0
    matrix[1][1] = 3.0
    matrix[1][2] = -6.0
    return matrix


def _mapping():
    core0 = hard_core(
        _producer(), threshold=THETA,
        sources=[SpikeSource(-2, i, is_input=True) for i in range(AXONS)],
    )
    core1 = hard_core(
        _consumer(), threshold=THETA,
        sources=[SpikeSource(PRODUCER, 0), SpikeSource(PRODUCER, 1)]
        + [SpikeSource(-1, 0, is_off=True)] * (AXONS - 2),
    )
    return mapping_of([core0, core1], [SpikeSource(CONSUMER, 0)])


def _rasters():
    """Cycle 0 drives BOTH the +3 and the -7 row: the window is net-negative.

    It also drives rows 2 AND 3 together, which is the window in which neuron 1
    receives two suprathreshold events at once.
    """
    def row(*active):
        line = [0] * AXONS
        for index in active:
            line[index] = 1
        return line

    negative = [
        row(0, 1, 2, 3, AXONS - 1),  # neuron 0: +3 - 7 = -4; neuron 1: +5 +5
        row(0),                     # -1
        row(0),                     # +2
        row(0),                     # +5 -> the first crossing
        row(0, 1),                  # -4 again
    ]
    quiet = [row(2), row(0, 1), row(0), row(0), row(0)]
    return [negative, quiet]


@pytest.fixture(scope="module")
def sync_fire():
    require_simulator()
    law = sync_fire_law(MEMBRANE_BITS)
    spec = spec_for(law, axons=AXONS, neurons=NEURONS, count=2)
    mapping = _mapping()
    rasters = _rasters()
    samples = traces_for(mapping, rasters, soma_law=law, simulation_length=S)
    cycles = len(samples[0].trace.outputs)
    bounds = require_no_saturation(
        mapping, spec=spec, thetas=thetas_of(mapping), cycles=cycles,
        membrane_init=0)
    generated = generate_core(spec, saturation_bounds=bounds)
    images = images_for(mapping, spec)
    with timed("256x256 sb16 sync-fire cosim"):
        result = run_variant_cosim(
            generated, images, [list(s.per_cycle) for s in samples],
            latencies=samples[0].trace.latencies)
    report("256x256sb16", result)
    print(f"[odin-gen] saturation bounds: "
          f"{[(b.core_index, b.lowest, b.highest) for b in bounds]} "
          f"register=[{spec.membrane_low}, {spec.membrane_high}]")
    return spec, mapping, rasters, samples, bounds, generated, result


class TestTheSyncFireRtlReproducesThePerCycleLaw:
    def test_no_cycle_of_any_sample_differs(self, sync_fire):
        _spec, _map, _rast, samples, _b, _gen, result = sync_fire
        assert compare_cycle_counts(result, samples) == []

    def test_the_generated_rtl_reports_the_spec_the_harness_believes(self, sync_fire):
        _spec, _map, _rast, _samples, _b, _gen, result = sync_fire
        assert result.capture.spec_failures == 0
        assert result.capture.spec_checks == result.plan.n_cores

    def test_no_rail_was_touched_on_the_silicon_either(self, sync_fire):
        """The static bound said the rails were unreachable; the RTL's own
        sticky flag is the runtime half of that same claim."""
        _spec, _map, _rast, _samples, _b, _gen, result = sync_fire
        assert result.capture.rail_failures == 0
        assert result.capture.rail_checks == result.plan.n_cores

    def test_the_fixture_is_not_vacuous_for_the_law_under_test(self, sync_fire):
        """Non-vacuity, on the INPUTS: without a neuron that receives two
        suprathreshold events in one cycle the two laws coincide everywhere and
        the cosimulation cannot discriminate them, whatever it reports."""
        _spec, mapping, _rast, samples, _b, _gen, _result = sync_fire
        assert suprathreshold_multiplicity(mapping, samples) >= 2

    def test_at_most_one_spike_per_neuron_per_cycle(self, sync_fire):
        """The law's signature, read off the RTL: a per-CYCLE core compares
        once, so the wire never carries two spikes from one neuron in one
        cycle — an event-serial core would, on this fixture, at cycle 0."""
        _spec, mapping, _rast, _samples, _b, _gen, result = sync_fire
        assert rtl_cycle_multiplicity(
            result, [int(core.neurons_per_core) for core in mapping.cores]) == 1
        assert result.capture.events


class TestTheNetNegativeCycleIsTheWholePoint:
    def test_the_signed_register_matches_the_unbounded_contract_exactly(
            self, sync_fire):
        _spec, mapping, rasters, samples, _b, _gen, _result = sync_fire
        for index, raster in enumerate(rasters):
            exact = simulate_cycles(
                mapping, soma_law=unbounded_law(), input_counts=raster,
                simulation_length=S)
            assert exact.outputs == samples[index].trace.outputs, index

    def test_an_unsigned_register_of_the_same_width_would_differ(self, sync_fire):
        """Teeth: the floor-at-zero register fires EARLY on the same fixture, so
        an implementation that quietly floored would fail this gate."""
        _spec, mapping, rasters, samples, _b, _gen, _result = sync_fire
        unsigned = simulate_cycles(
            mapping,
            soma_law=SomaLaw(
                firing_mode="Novena", thresholding_mode="<=",
                firing_granularity="per_cycle",
                membrane_arithmetic="saturating_unsigned",
                membrane_bits=MEMBRANE_BITS),
            input_counts=rasters[0], simulation_length=S)
        assert unsigned.outputs != samples[0].trace.outputs

    def test_the_witness_neuron_really_goes_negative_and_fires_late(self, sync_fire):
        _spec, _map, _rast, _samples, _b, _gen, result = sync_fire
        fired = [
            result.cycle_counts(0, cycle, PRODUCER, 1)[0]
            for cycle in range(result.plan.cycles_per_sample)
        ]
        assert fired[:3] == [0, 0, 0], fired
        assert fired[3] == 1, fired

    def test_the_static_bound_proves_a_negative_floor(self, sync_fire):
        spec, _map, _rast, _samples, bounds, _gen, _result = sync_fire
        assert bounds, "the no-saturation gate did not run"
        assert min(bound.lowest for bound in bounds) < 0
        for bound in bounds:
            assert bound.inside(spec)


class TestNevresimIsTheThirdArm:
    @pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
    def test_nevresim_matches_the_torch_twin_and_therefore_the_rtl(self, sync_fire):
        spec, mapping, rasters, samples, _b, _gen, result = sync_fire
        ensure_nevresim_ready()
        from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

        loader = [
            (np.asarray(raster, dtype=np.float64).reshape(-1), np.zeros(NEURONS))
            for raster in rasters
        ]
        with timed("sync-fire nevresim"), tempfile.TemporaryDirectory() as tmp:
            driver = NevresimDriver(
                AXONS, mapping, tmp, int,
                spike_generation_mode="SpikeTrain", firing_mode="Novena",
                thresholding_mode="<=", spiking_mode="lif", threshold_type=int,
                connectivity_mode="runtime", verbose=False,
                soma_law=spec.soma_law)
            _raw, records = driver.predict_spiking_raw_with_records(
                loader, S, max(samples[0].trace.latencies))

        rtl = result.window_counts(
            latencies=samples[0].trace.latencies, simulation_length=S,
            neurons=[int(core.neurons_per_core) for core in mapping.cores])
        for sample, trace in enumerate(samples):
            fold = trace.trace.window_counts()
            for core in range(len(mapping.cores)):
                nevresim = tuple(
                    int(v) for v in np.asarray(records[sample][core]["out"])[:NEURONS])
                assert nevresim == fold[core], (sample, core)
                assert nevresim == rtl[sample][core], (sample, core)
