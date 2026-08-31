"""[ODIN6, plan §7 row 20] Geometry variants: the generated RTL IS the fold.

Two `CoreSpec`s that the vendored stock core cannot be — 128x128 on an 8-bit
membrane, and 512x256 on a 16-bit one — are generated from `hw/gen`, programmed
through the generated core's own configuration port, and compared against the
same per-cycle policy the deployed torch executors are built from, at ZERO
difference, cycle by cycle, over two consecutive samples.

The 16-bit variant carries the plan's §11 relief witness: its threshold is 300,
which the stock 8-bit membrane cannot hold at all (the theta-ceiling refusal is
exactly what the wider generated membrane exists to lift).
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
    per_event_law,
    report,
    require_simulator,
    spec_for,
    timed,
    traces_for,
)

from integration.parity_harness import ensure_nevresim_ready, have_cxx_compiler

from mimarsinan.chip_simulation.odin_rtl.gen_cosim import run_variant_cosim
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin_gen import (
    generate_core,
    is_stock_spec,
    stock_core_spec,
)
from mimarsinan.mapping.export.odin_gen.passthrough import vendored_file_set
from mimarsinan.mapping.export.odin_gen.variants import variant_named
from mimarsinan.models.spiking.serial.refusals import (
    EMISSION_COUNT_CEILING,
    count_ceiling,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]

STOCK_MEMBRANE_CEILING = 255


# ---------------------------------------------------------------------------
# Variant 1 — 128 x 128, 8-bit unsigned membrane, two cores
# ---------------------------------------------------------------------------

#: The geometry is READ from the catalog the compile-limits study costs, so a
#: variant can never be measured at a geometry no cosimulation proved.
SMALL_VARIANT = variant_named("gen_a128n128_mb8_per_event")
SMALL_AXONS = SMALL_VARIANT.spec.max_axons
SMALL_NEURONS = SMALL_VARIANT.spec.max_neurons
SMALL_S = 3
PRODUCER = 0
CONSUMER = 1


def _small_mapping():
    """Sparse weights over a DENSE axon raster, so every row address is driven."""
    producer = np.zeros((SMALL_AXONS, SMALL_NEURONS), dtype=np.float64)
    producer[0][0] = 5.0
    producer[0][1] = 1.0
    producer[64][2] = 3.0
    producer[64][3] = -2.0
    producer[65][2] = -3.0
    producer[65][3] = 5.0
    # The multiplicity producer sits on the TOP three rows of the widened
    # crossbar: one input cycle leaves neuron 4 as THREE spikes.
    for row in (125, 126, 127):
        producer[row][4] = 5.0
    producer[10][SMALL_NEURONS - 1] = 4.0

    consumer = np.zeros((SMALL_AXONS, SMALL_NEURONS), dtype=np.float64)
    consumer[0][0] = 5.0
    consumer[1][1] = 3.0
    consumer[2][1] = -3.0

    core0 = hard_core(
        producer, threshold=4.0,
        sources=[SpikeSource(-2, i, is_input=True) for i in range(SMALL_AXONS)],
    )
    core1 = hard_core(
        consumer, threshold=5.0,
        sources=[SpikeSource(PRODUCER, 4)] * 3
        + [SpikeSource(-1, 0, is_off=True)] * (SMALL_AXONS - 3),
    )
    return mapping_of(
        [core0, core1],
        [SpikeSource(CONSUMER, 0), SpikeSource(CONSUMER, 1)],
    )


def _small_rasters():
    """Every input line but one is driven, so the whole axon range is addressed."""
    row = [1] * SMALL_AXONS
    row[1] = 0
    alternating = [1 if index % 3 else 0 for index in range(SMALL_AXONS)]
    return [
        [list(row) for _ in range(SMALL_S)],
        [list(alternating) for _ in range(SMALL_S)],
    ]


@pytest.fixture(scope="module")
def small_variant():
    require_simulator()
    law = SMALL_VARIANT.spec.soma_law
    spec = spec_for(law, axons=SMALL_AXONS, neurons=SMALL_NEURONS, count=2)
    mapping = _small_mapping()
    generated = generate_core(spec)
    images = images_for(mapping, spec)
    samples = traces_for(
        mapping, _small_rasters(), soma_law=law, simulation_length=SMALL_S)
    with timed("128x128 mb8 per_event cosim"):
        result = run_variant_cosim(
            generated, images, [list(s.per_cycle) for s in samples],
            latencies=samples[0].trace.latencies)
    report("128x128mb8", result)
    return spec, mapping, generated, samples, result


class TestTheSmallGeometryVariantReproducesTheFold:
    def test_no_cycle_of_any_sample_differs(self, small_variant):
        _spec, _map, _gen, samples, result = small_variant
        assert compare_cycle_counts(result, samples) == []

    def test_the_generated_rtl_reports_the_spec_the_harness_believes(
            self, small_variant):
        _spec, _map, _gen, _samples, result = small_variant
        assert result.capture.spec_failures == 0
        assert result.capture.spec_checks == result.plan.n_cores

    def test_the_comparison_is_not_vacuous(self, small_variant):
        _spec, _map, _gen, samples, result = small_variant
        assert result.capture.events, "the RTL produced no output events at all"
        emitted = [
            max(counts) for sample in samples
            for per_core in sample.trace.outputs for counts in per_core
        ]
        assert max(emitted) >= 3, (
            "no multiplicity on the wire: a per-cycle law would reproduce this "
            "fixture and the gate would prove nothing")

    def test_the_sign_asymmetric_neuron_witnesses_adjacency(self, small_variant):
        """w=(+3,-3) on two slots carrying the same count 3 with theta 5:
        adjacent occurrences cross once, a round-robin drain never crosses."""
        _spec, _map, _gen, _samples, result = small_variant
        witness = [
            result.cycle_counts(0, cycle, CONSUMER, 2)[1]
            for cycle in range(1, result.plan.cycles_per_sample)
        ]
        assert witness == [1] * (result.plan.cycles_per_sample - 1), witness

    def test_the_widened_crossbar_is_actually_used(self, small_variant):
        spec, mapping, _gen, _samples, _result = small_variant
        assert spec.max_axons == SMALL_AXONS
        assert int(mapping.cores[0].axons_per_core) == SMALL_AXONS
        assert float(np.asarray(mapping.cores[0].core_matrix)[127][4]) == 5.0


# ---------------------------------------------------------------------------
# Variant 2 — 512 x 256, 16-bit unsigned membrane, one core, theta over the
# stock ceiling
# ---------------------------------------------------------------------------

WIDE_VARIANT = variant_named("gen_a512n256_mb16_per_event")
WIDE_AXONS = WIDE_VARIANT.spec.max_axons
WIDE_NEURONS = WIDE_VARIANT.spec.max_neurons
WIDE_S = 2
WIDE_THETA = 300.0


def _wide_mapping():
    matrix = np.zeros((WIDE_AXONS, WIDE_NEURONS), dtype=np.float64)
    # 100 rows x 7 = 700 charge against theta=300: TWO spikes in one cycle,
    # which no per-cycle law can produce.
    for row in range(100):
        matrix[row][0] = 7.0
    # 50 rows x 7 = 350: exactly one crossing, on the LAST neuron address.
    for row in range(400, 450):
        matrix[row][WIDE_NEURONS - 1] = 7.0
    # A neuron that floors instead of firing, driven from the last axon row.
    for row in (300, 301, 302):
        matrix[row][1] = 7.0
    matrix[WIDE_AXONS - 1][1] = -7.0
    core = hard_core(
        matrix, threshold=WIDE_THETA,
        sources=[SpikeSource(-2, i, is_input=True) for i in range(WIDE_AXONS)],
    )
    return mapping_of([core], [SpikeSource(0, 0)])


def _wide_raster():
    row = [0] * WIDE_AXONS
    for index in list(range(100)) + [300, 301, 302] + list(range(400, 450)) + [511]:
        row[index] = 1
    return [list(row) for _ in range(WIDE_S)]


@pytest.fixture(scope="module")
def wide_variant():
    require_simulator()
    law = WIDE_VARIANT.spec.soma_law
    spec = spec_for(law, axons=WIDE_AXONS, neurons=WIDE_NEURONS)
    mapping = _wide_mapping()
    generated = generate_core(spec)
    images = images_for(mapping, spec)
    samples = traces_for(
        mapping, [_wide_raster(), _wide_raster()], soma_law=law,
        simulation_length=WIDE_S)
    with timed("512x256 mb16 per_event cosim"):
        result = run_variant_cosim(
            generated, images, [list(s.per_cycle) for s in samples],
            latencies=samples[0].trace.latencies)
    report("512x256mb16", result)
    return spec, mapping, generated, samples, result


class TestTheWideGeometryVariantReproducesTheFold:
    def test_no_cycle_of_any_sample_differs(self, wide_variant):
        _spec, _map, _gen, samples, result = wide_variant
        assert compare_cycle_counts(result, samples) == []

    def test_the_generated_rtl_reports_the_spec_the_harness_believes(
            self, wide_variant):
        _spec, _map, _gen, _samples, result = wide_variant
        assert result.capture.spec_failures == 0

    def test_the_threshold_is_over_the_stock_membrane_ceiling(self, wide_variant):
        """The plan's §11 relief, measured: this deployment is refused outright
        on the stock 8-bit membrane and runs on the generated 16-bit one."""
        spec, _map, _gen, _samples, _result = wide_variant
        assert WIDE_THETA > STOCK_MEMBRANE_CEILING
        assert spec.membrane_high == 65535

    def test_the_wide_variant_still_carries_multiplicity(self, wide_variant):
        _spec, _map, _gen, samples, result = wide_variant
        assert result.cycle_counts(0, 0, 0, 2)[0] == 2
        emitted = [
            max(counts) for sample in samples
            for per_core in sample.trace.outputs for counts in per_core
        ]
        assert max(emitted) >= 2

    def test_the_highest_neuron_and_axon_addresses_are_exercised(
            self, wide_variant):
        _spec, mapping, _gen, _samples, result = wide_variant
        assert result.cycle_counts(0, 0, 0, WIDE_NEURONS)[WIDE_NEURONS - 1] == 1
        assert float(
            np.asarray(mapping.cores[0].core_matrix)[WIDE_AXONS - 1][1]) == -7.0


# ---------------------------------------------------------------------------
# Variant 3 — the WIDE CHIP CONFIG: 1024 x 256, 16-bit unsigned membrane, an
# 8-bit signed synapse cell. Both axes the stock geometry refuses move at once.
# ---------------------------------------------------------------------------

WIDE_CHIP_VARIANT = variant_named("gen_a1024n256_mb16w8_per_event")
WIDE_CHIP_AXONS = WIDE_CHIP_VARIANT.spec.max_axons
WIDE_CHIP_NEURONS = WIDE_CHIP_VARIANT.spec.max_neurons
WIDE_CHIP_S = 2
WIDE_CHIP_THETA = 300.0

#: The stock crossbar stores an unsigned magnitude signed once per PHYSICAL ROW,
#: so its representable weight set at 4 bits is the SYMMETRIC [-7, 7]. Every
#: weight in the fixture below is outside it.
STOCK_WEIGHT_CEILING = 7

#: A 784-line MNIST raster is what 1024 rows exist for; the fixture drives the
#: addresses that decide whether the widened row address is real.
MNIST_INPUT_LINES = 784


def _wide_chip_mapping():
    matrix = np.zeros((WIDE_CHIP_AXONS, WIDE_CHIP_NEURONS), dtype=np.float64)
    # Six rows x 100 = 600 against theta=300: TWO crossings in one cycle, on a
    # weight the stock cell cannot hold at all.
    for row in range(6):
        matrix[row][0] = 100.0
    # A neuron driven from ABOVE the 784th row -- the addresses that only exist
    # because the row address widened -- and floored by the extreme NEGATIVE
    # weight, which a per-axon-signed magnitude cell has no encoding for.
    for row in (900, 901, 902):
        matrix[row][1] = 100.0
    matrix[WIDE_CHIP_AXONS - 1][1] = -128.0
    # The LAST neuron address, driven from rows straddling the MNIST width.
    for row in (MNIST_INPUT_LINES - 1, MNIST_INPUT_LINES, MNIST_INPUT_LINES + 1):
        matrix[row][WIDE_CHIP_NEURONS - 1] = 100.0
    # The top of the two's-complement cell, three times, one short of theta.
    for row in (10, 11):
        matrix[row][2] = 127.0
    core = hard_core(
        matrix, threshold=WIDE_CHIP_THETA,
        sources=[SpikeSource(-2, i, is_input=True) for i in range(WIDE_CHIP_AXONS)],
    )
    return mapping_of([core], [SpikeSource(0, 0)])


def _wide_chip_raster():
    row = [0] * WIDE_CHIP_AXONS
    driven = (list(range(6)) + [10, 11] + [900, 901, 902]
              + [MNIST_INPUT_LINES - 1, MNIST_INPUT_LINES, MNIST_INPUT_LINES + 1]
              + [WIDE_CHIP_AXONS - 1])
    for index in driven:
        row[index] = 1
    return [list(row) for _ in range(WIDE_CHIP_S)]


@pytest.fixture(scope="module")
def wide_chip_variant():
    require_simulator()
    spec = WIDE_CHIP_VARIANT.spec
    law = spec.soma_law
    mapping = _wide_chip_mapping()
    generated = generate_core(spec)
    images = images_for(mapping, spec)
    samples = traces_for(
        mapping, [_wide_chip_raster(), _wide_chip_raster()], soma_law=law,
        simulation_length=WIDE_CHIP_S)
    with timed("1024x256 mb16 w8 per_event cosim"):
        result = run_variant_cosim(
            generated, images, [list(s.per_cycle) for s in samples],
            latencies=samples[0].trace.latencies)
    report("1024x256mb16w8", result)
    return spec, mapping, generated, samples, result


class TestTheWideChipConfigVariantReproducesTheFold:
    """The WIDE chip configuration's core, proved the same way as the others."""

    def test_no_cycle_of_any_sample_differs(self, wide_chip_variant):
        _spec, _map, _gen, samples, result = wide_chip_variant
        assert compare_cycle_counts(result, samples) == []

    def test_the_generated_rtl_reports_the_spec_the_harness_believes(
            self, wide_chip_variant):
        _spec, _map, _gen, _samples, result = wide_chip_variant
        assert result.capture.spec_failures == 0
        assert result.capture.spec_checks == result.plan.n_cores

    def test_a_whole_mnist_raster_maps_onto_one_core(self, wide_chip_variant):
        """The reason 1024 rows exist: 784 input lines, one slot each, and the
        always-on bias row still has somewhere to go."""
        spec, _map, _gen, _samples, _result = wide_chip_variant
        assert spec.max_axons >= MNIST_INPUT_LINES + 1
        assert spec.max_axons == WIDE_CHIP_AXONS

    def test_the_weights_are_outside_the_stock_cell_entirely(
            self, wide_chip_variant):
        """The other half of the relief: the stock cell holds an unsigned
        magnitude signed per row, so [-7, 7] is all it represents."""
        spec, mapping, _gen, _samples, _result = wide_chip_variant
        matrix = np.asarray(mapping.cores[0].core_matrix)
        assert float(matrix.max()) > STOCK_WEIGHT_CEILING
        assert float(matrix.min()) < -STOCK_WEIGHT_CEILING
        # The extreme negative cell of a two's-complement nibble-times-two has
        # no per-axon-signed encoding at all.
        assert float(matrix.min()) == -(1 << (spec.weight_bits - 1))
        assert spec.weight_bits == 8

    def test_the_threshold_is_over_the_stock_membrane_ceiling(
            self, wide_chip_variant):
        spec, _map, _gen, _samples, _result = wide_chip_variant
        assert WIDE_CHIP_THETA > STOCK_MEMBRANE_CEILING
        assert spec.membrane_high == 65535

    def test_the_wide_chip_variant_carries_multiplicity(self, wide_chip_variant):
        _spec, _map, _gen, samples, result = wide_chip_variant
        assert result.cycle_counts(0, 0, 0, 3)[0] == 2
        emitted = [
            max(counts) for sample in samples
            for per_core in sample.trace.outputs for counts in per_core
        ]
        assert max(emitted) >= 2

    def test_the_highest_row_and_neuron_addresses_are_exercised(
            self, wide_chip_variant):
        _spec, mapping, _gen, _samples, result = wide_chip_variant
        counts = result.cycle_counts(0, 0, 0, WIDE_CHIP_NEURONS)
        assert counts[WIDE_CHIP_NEURONS - 1] == 1
        assert float(np.asarray(mapping.cores[0].core_matrix)
                     [WIDE_CHIP_AXONS - 1][1]) == -128.0


class TestNevresimIsTheThirdArmOnTheSmallVariant:
    @pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
    def test_nevresim_matches_the_torch_twin_and_therefore_the_rtl(
            self, small_variant):
        _spec, mapping, _gen, samples, result = small_variant
        ensure_nevresim_ready()
        from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

        law = per_event_law(8)
        loader = [
            (np.asarray(raster, dtype=np.float64).reshape(-1),
             np.zeros(SMALL_NEURONS))
            for raster in _small_rasters()
        ]
        with timed("128x128 nevresim"), tempfile.TemporaryDirectory() as tmp:
            driver = NevresimDriver(
                SMALL_AXONS, mapping, tmp, int,
                spike_generation_mode="SpikeTrain", firing_mode="Novena",
                thresholding_mode="<=", spiking_mode="lif", threshold_type=int,
                connectivity_mode="runtime", verbose=False, soma_law=law)
            _raw, records = driver.predict_spiking_raw_with_records(
                loader, SMALL_S, max(samples[0].trace.latencies))

        rtl = result.window_counts(
            latencies=samples[0].trace.latencies, simulation_length=SMALL_S,
            neurons=[int(core.neurons_per_core) for core in mapping.cores])
        for sample, trace in enumerate(samples):
            fold = trace.trace.window_counts()
            for core in range(len(mapping.cores)):
                nevresim = tuple(
                    int(v) for v in
                    np.asarray(records[sample][core]["out"])[:SMALL_NEURONS])
                assert nevresim == fold[core], (sample, core)
                assert nevresim == rtl[sample][core], (sample, core)


class TestTheStockSpecStaysAVendoredPassthrough:
    """Plan §7 row 20's other half, re-asserted inside the named runner: the
    out-of-the-box core is the vendored tree and generation never rewrites it."""

    def test_generating_the_stock_spec_reproduces_the_vendored_tree(self):
        spec = stock_core_spec()
        assert is_stock_spec(spec)
        generated = generate_core(spec)
        assert generated.vendored is True
        assert dict(generated.files) == dict(vendored_file_set())
        print(f"[odin-gen] stock passthrough: {len(generated.files)} vendored "
              f"files byte-identical, spec_key={spec.spec_key()}")


# ---------------------------------------------------------------------------
# [ODIN C4] The COUNT CURRENCY at the chip's own width — the wall C3 measured
# ---------------------------------------------------------------------------

#: 300 spikes from ONE neuron in ONE cycle. Every implementation refused this at
#: 127 before the currency became a chip-claimed quantity; on a chip declaring a
#: 16-bit register all three arms must now carry it EXACTLY, unclamped. 300 and
#: not 200 on purpose: it is past an UNSIGNED byte too, so a raster that stored
#: counts in one would wrap to 44 rather than survive by luck.
HIGH_MULTIPLICITY = 300
HIGH_MULT_S = 2
HIGH_MULT_THETA = 5.0
HIGH_MULT_TAIL = 3


def _high_multiplicity_mapping():
    """theta=5 with a weight of 5 on slot 0: every occurrence crosses and the
    hard-zero reset returns the membrane, so the emitted count IS the slot's
    multiplicity — the cleanest hand-derived number a fold can be checked on."""
    matrix = np.zeros((WIDE_AXONS, WIDE_NEURONS), dtype=np.float64)
    matrix[0][0] = 5.0
    # A neuron that needs TWO occurrences of its slot to cross once, so the
    # case is not one saturated neuron and nothing else.
    matrix[5][1] = 3.0
    # The last neuron address, driven from the last axon row.
    matrix[WIDE_AXONS - 1][WIDE_NEURONS - 1] = 5.0
    core = hard_core(
        matrix, threshold=HIGH_MULT_THETA,
        sources=[SpikeSource(-2, i, is_input=True) for i in range(WIDE_AXONS)],
    )
    return mapping_of([core], [SpikeSource(0, 0)])


def _high_multiplicity_raster():
    first = [0] * WIDE_AXONS
    first[0] = HIGH_MULTIPLICITY
    first[5] = 2
    first[WIDE_AXONS - 1] = 1
    second = [0] * WIDE_AXONS
    second[0] = HIGH_MULT_TAIL
    return [first, second]


@pytest.fixture(scope="module")
def high_multiplicity_variant():
    require_simulator()
    law = per_event_law(16)
    spec = spec_for(law, axons=WIDE_AXONS, neurons=WIDE_NEURONS)
    mapping = _high_multiplicity_mapping()
    generated = generate_core(spec)
    images = images_for(mapping, spec)
    samples = traces_for(
        mapping, [_high_multiplicity_raster()], soma_law=law,
        simulation_length=HIGH_MULT_S)
    with timed("512x256 mb16 high-multiplicity cosim"):
        result = run_variant_cosim(
            generated, images, [list(s.per_cycle) for s in samples],
            latencies=samples[0].trace.latencies)
    report("512x256mb16-highmult", result)
    return spec, mapping, samples, result


class TestTheCountCurrencyIsCarriedAtTheChipsOwnWidth:
    """[ODIN C4] The wall Phase C3 measured, discharged on three independent
    arms: 200 events in one cycle, exact everywhere, refused nowhere."""

    def test_the_case_is_genuinely_over_the_stock_currency(
            self, high_multiplicity_variant):
        spec, _mapping, samples, _result = high_multiplicity_variant
        peak = max(
            int(count) for sample in samples
            for per_core in sample.trace.outputs for counts in per_core
            for count in counts)
        assert peak == HIGH_MULTIPLICITY
        assert peak > EMISSION_COUNT_CEILING
        assert peak > 255  # and past an unsigned byte, so no store survives by luck
        assert peak <= count_ceiling(spec)
        assert count_ceiling(spec) == 32767

    def test_no_cycle_of_the_rtl_differs_from_the_fold(
            self, high_multiplicity_variant):
        _spec, _mapping, samples, result = high_multiplicity_variant
        assert compare_cycle_counts(result, samples) == []
        assert result.capture.spec_failures == 0

    def test_the_rtl_emitted_every_transaction_of_the_cycle(
            self, high_multiplicity_variant):
        """The wire carries no count field: 300 is 300 ADJACENT AER
        transactions, which is what makes the currency a representation."""
        _spec, _mapping, _samples, result = high_multiplicity_variant
        assert result.cycle_counts(0, 0, 0, 2)[0] == HIGH_MULTIPLICITY
        assert result.cycle_counts(0, 1, 0, 1)[0] == HIGH_MULT_TAIL

    @pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
    def test_nevresim_carries_the_same_count_as_the_fold_and_the_rtl(
            self, high_multiplicity_variant):
        _spec, mapping, samples, result = high_multiplicity_variant
        ensure_nevresim_ready()
        from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

        loader = [
            (np.asarray(raster, dtype=np.float64).reshape(-1),
             np.zeros(WIDE_NEURONS))
            for raster in [_high_multiplicity_raster()]
        ]
        with timed("512x256 mb16 high-multiplicity nevresim"), \
                tempfile.TemporaryDirectory() as tmp:
            driver = NevresimDriver(
                WIDE_AXONS, mapping, tmp, int,
                spike_generation_mode="SpikeTrain", firing_mode="Novena",
                thresholding_mode="<=", spiking_mode="lif", threshold_type=int,
                connectivity_mode="runtime", verbose=False,
                soma_law=per_event_law(16))
            _raw, records = driver.predict_spiking_raw_with_records(
                loader, HIGH_MULT_S, max(samples[0].trace.latencies))

        rtl = result.window_counts(
            latencies=samples[0].trace.latencies,
            simulation_length=HIGH_MULT_S,
            neurons=[int(core.neurons_per_core) for core in mapping.cores])
        fold = samples[0].trace.window_counts()
        nevresim = tuple(
            int(v) for v in np.asarray(records[0][0]["out"])[:WIDE_NEURONS])
        assert nevresim == fold[0]
        assert nevresim == rtl[0][0]
        # The hand-derived number, stated once so a self-consistent trio of
        # wrong implementations still fails: 300 crossings then 3 more.
        assert nevresim[0] == HIGH_MULTIPLICITY + HIGH_MULT_TAIL
        assert nevresim[1] == 1
        assert nevresim[WIDE_NEURONS - 1] == 1

    @pytest.mark.skipif(not have_cxx_compiler(), reason="C++ compiler unavailable")
    def test_the_counted_raster_carries_the_per_cycle_count_too(
            self, high_multiplicity_variant):
        """SPKTRN2, not just SPKREC: the per-CYCLE raster nevresim prints and
        the host re-packs is the other half of the currency, and a byte-wide
        store would have wrapped 200 to 200 only by luck and 300 to 44."""
        _spec, mapping, samples, _result = high_multiplicity_variant
        ensure_nevresim_ready()
        from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver

        loader = [
            (np.asarray(_high_multiplicity_raster(), dtype=np.float64).reshape(-1),
             np.zeros(WIDE_NEURONS))
        ]
        with tempfile.TemporaryDirectory() as tmp:
            driver = NevresimDriver(
                WIDE_AXONS, mapping, tmp, int,
                spike_generation_mode="SpikeTrain", firing_mode="Novena",
                thresholding_mode="<=", spiking_mode="lif", threshold_type=int,
                connectivity_mode="runtime", verbose=False,
                soma_law=per_event_law(16))
            _raw, _records, trains = driver.predict_spiking_raw_with_spike_trains(
                loader, HIGH_MULT_S, max(samples[0].trace.latencies))

        raster = trains[0][0]
        assert int(raster[0][0]) == HIGH_MULTIPLICITY
        assert int(raster[0][1]) == HIGH_MULT_TAIL
        # The identity the two record builds owe each other, at a count no byte
        # holds: the raster's window sum IS the SPKREC count.
        assert int(raster[0].sum()) == HIGH_MULTIPLICITY + HIGH_MULT_TAIL

    def test_the_same_events_are_refused_by_an_eight_bit_register(self):
        """Non-vacuity: nothing about this case is legal on the stock currency."""
        import torch

        from mimarsinan.models.spiking.serial import (
            EmissionBoundExceededError,
            lif_serial_fold,
        )

        events = torch.tensor([[float(HIGH_MULTIPLICITY)]], dtype=torch.float64)
        weight = torch.tensor([[5.0]], dtype=torch.float64)
        theta = torch.tensor(HIGH_MULT_THETA, dtype=torch.float64)
        for bits, expectation in ((8, pytest.raises(EmissionBoundExceededError)),
                                  (16, None)):
            memb = torch.zeros(1, 1, dtype=torch.float64)
            if expectation is None:
                counts = lif_serial_fold(
                    memb, weight, events, theta, soma_law=per_event_law(bits))
                assert counts.tolist() == [[float(HIGH_MULTIPLICITY)]]
            else:
                with expectation:
                    lif_serial_fold(memb, weight, events, theta,
                                    soma_law=per_event_law(bits))
