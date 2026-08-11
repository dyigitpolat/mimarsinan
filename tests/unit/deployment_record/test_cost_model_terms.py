"""The cost model's terms over the fixture record: exact arithmetic + epistemics.

Every number here is recomputed from the SAME sources the model is allowed to
use — the imported ``weight_reuse_cost_model`` coefficients and the SANA-FE
per-event presets — with only the RECORD quantities (196/292/484 payload
bytes, 3 cores, 1 barrier) written as literals: those are the model's own
arithmetic and must not be free to drift.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mimarsinan.chip_simulation.sanafe.presets import LOIHI_PRESET, TRUENORTH_PRESET
from mimarsinan.chip_simulation.weight_reuse_cost_model import DEFAULT_COEFFICIENT_BAND
from mimarsinan.deployment_record.cost import (
    CORE_INIT,
    CostTerm,
    DeploymentCostModel,
    DeploymentCostReport,
    find_term,
)
from mimarsinan.deployment_record.cost.coefficients import (
    PROGRAMMING_BANDWIDTH_BYTES_PER_S,
    SYNC_BARRIER_S,
)
from mimarsinan.deployment_record.schema import Band
from unit.deployment_record.record_fixtures import make_full_record

# The fixture record's own quantities (record_fixtures.make_schedule).
REPROGRAM_CORES = 2
RESIDENT_CORES = 1
TOTAL_CORES = REPROGRAM_CORES + RESIDENT_CORES
SYNC_COUNT = 1
COMPUTE_S = 3e-3
# params_bytes=100 + connectivity_entries=24 x (4 / 8 / 16) bytes per entry.
PAYLOAD_BYTES = (196.0, 292.0, 484.0)
CORNERS = ("low", "nominal", "high")

MODEL = DeploymentCostModel()


def band_values(band: Band):
    return (band.low, band.nominal, band.high)


def term_band(term: CostTerm):
    assert term.band is not None, f"{term.name} should be banded"
    return band_values(term.band)


def dma_corners():
    return tuple(
        getattr(DEFAULT_COEFFICIENT_BAND, corner).e_dma_per_byte_mj for corner in CORNERS
    )


def sync_corners():
    return tuple(
        getattr(DEFAULT_COEFFICIENT_BAND, corner).e_sync_barrier_mj for corner in CORNERS
    )


def preset_core_init(access_key: str, update_key: str, scale: float):
    """The §7 basis recomputed here: per-neuron soma reset x a reference core."""
    truenorth = TRUENORTH_PRESET[access_key] + TRUENORTH_PRESET[update_key]
    loihi = LOIHI_PRESET[access_key] + LOIHI_PRESET[update_key]
    return (truenorth * 256 * scale, loihi * 256 * scale, loihi * 1024 * scale)


CORE_INIT_ENERGY = preset_core_init(
    "soma_access_energy_j", "soma_update_energy_j", 1000.0
)
CORE_INIT_TIME = preset_core_init("soma_access_latency_s", "soma_update_latency_s", 1.0)


@pytest.fixture
def record():
    return make_full_record()


@pytest.fixture
def report(record) -> DeploymentCostReport:
    return MODEL.evaluate(record)


def reprogram_cost(record):
    costs = MODEL.segment_initialization(record)
    assert [cost.programming for cost in costs] == ["reprogram", "resident"]
    return costs[0]


def resident_cost(record):
    return MODEL.segment_initialization(record)[1]


# ── segment initialization ──────────────────────────────────────────────


def test_segment_payload_bytes_are_exact_counts_at_a_modeled_width(record):
    assert term_band(reprogram_cost(record).term("payload_bytes")) == PAYLOAD_BYTES


def test_segment_payload_energy_applies_the_imported_dma_band(record):
    expected = tuple(
        payload * e_dma for payload, e_dma in zip(PAYLOAD_BYTES, dma_corners())
    )
    assert term_band(reprogram_cost(record).term("payload_energy_mj")) == expected


def test_segment_reset_is_core_count_times_the_preset_derived_constant(record):
    cost = reprogram_cost(record)
    assert cost.core_count == REPROGRAM_CORES
    assert term_band(cost.term("reset_energy_mj")) == pytest.approx(
        tuple(REPROGRAM_CORES * value for value in CORE_INIT_ENERGY), rel=1e-12
    )
    assert term_band(cost.term("reset_time_s")) == pytest.approx(
        tuple(REPROGRAM_CORES * value for value in CORE_INIT_TIME), rel=1e-12
    )


def test_segment_programming_time_divides_by_the_opposite_bandwidth_corner(record):
    bandwidth = band_values(PROGRAMMING_BANDWIDTH_BYTES_PER_S)
    expected = (
        PAYLOAD_BYTES[0] / bandwidth[2],
        PAYLOAD_BYTES[1] / bandwidth[1],
        PAYLOAD_BYTES[2] / bandwidth[0],
    )
    assert term_band(reprogram_cost(record).term("programming_time_s")) == expected


def test_resident_segment_costs_exactly_the_reset_constant(record):
    cost = resident_cost(record)
    assert cost.core_count == RESIDENT_CORES
    for name in ("payload_bytes", "payload_energy_mj", "programming_time_s"):
        assert term_band(cost.term(name)) == (0.0, 0.0, 0.0), name
    assert term_band(cost.term("reset_energy_mj")) == pytest.approx(
        tuple(RESIDENT_CORES * value for value in CORE_INIT_ENERGY), rel=1e-12
    )
    assert term_band(cost.term("reset_time_s")) == pytest.approx(
        tuple(RESIDENT_CORES * value for value in CORE_INIT_TIME), rel=1e-12
    )


def test_the_connectivity_byte_width_is_a_model_parameter(record):
    one_byte = DeploymentCostModel(
        bytes_per_connectivity_entry=Band(1.0, 1.0, 1.0, basis="test override")
    )
    cost = one_byte.segment_initialization(record)[0]
    assert term_band(cost.term("payload_bytes")) == (124.0, 124.0, 124.0)


# ── energy ──────────────────────────────────────────────────────────────


def test_energy_passes_measured_terms_through_unbanded(record):
    terms = MODEL.energy(record)
    assert find_term(terms, "measured_total_mj").value == 4.0
    assert find_term(terms, "measured_sanafe_total_mj").value == 4.0
    for name in ("measured_total_mj", "measured_sanafe_total_mj"):
        assert find_term(terms, name).band is None
        assert find_term(terms, name).kind == "measured"


def test_energy_never_passes_a_record_modeled_term_through(record):
    """The fixture record carries a modeled ``programming`` term: re-derived, not reused."""
    assert any(t.kind == "modeled" for t in record.energy.breakdown)
    names = [term.name for term in MODEL.energy(record)]
    assert "measured_programming_mj" not in names
    assert "modeled_programming_mj" in names


def test_energy_modeled_terms_are_the_record_quantities_times_the_bands(record):
    terms = MODEL.energy(record)
    expected_payload = tuple(
        payload * e_dma for payload, e_dma in zip(PAYLOAD_BYTES, dma_corners())
    )
    assert term_band(find_term(terms, "modeled_programming_mj")) == expected_payload
    assert term_band(find_term(terms, "modeled_core_init_mj")) == pytest.approx(
        tuple(TOTAL_CORES * value for value in CORE_INIT_ENERGY), rel=1e-12
    )
    assert term_band(find_term(terms, "modeled_sync_mj")) == pytest.approx(
        tuple(SYNC_COUNT * value for value in sync_corners()), rel=1e-12
    )


def test_energy_total_is_the_measurement_plus_the_modeled_terms(record):
    terms = MODEL.energy(record)
    modeled = [
        find_term(terms, name)
        for name in ("modeled_programming_mj", "modeled_core_init_mj", "modeled_sync_mj")
    ]
    total = find_term(terms, "total_mj")
    assert total.kind == "derived"
    for index, corner in enumerate(CORNERS):
        expected = 4.0 + sum(term_band(term)[index] for term in modeled)
        assert term_band(total)[index] == pytest.approx(expected, rel=1e-12)


def test_energy_requires_the_measured_energy_fragment(record):
    with pytest.raises(ValueError, match="energy fragment"):
        MODEL.energy(replace(record, energy=None))


# ── latency ─────────────────────────────────────────────────────────────


def test_latency_compute_is_measured_and_host_ops_appear_only_when_walled(record):
    terms = MODEL.latency(record)
    compute = find_term(terms, "compute_s")
    assert (compute.value, compute.band, compute.kind) == (COMPUTE_S, None, "measured")
    assert "host_ops_s" not in [term.name for term in terms]

    # A whole-run wall of 0.25 s over 5 traversals contributes 0.05 s to the
    # PER-SAMPLE decomposition; the raw total never joins unscaled.
    walled = replace(
        record,
        timing=replace(
            record.timing,
            latency=replace(
                record.timing.latency, host_ops_s=0.25, host_ops_s_per_pass=0.05
            ),
        ),
    )
    host_ops = find_term(MODEL.latency(walled), "host_ops_s")
    assert (host_ops.value, host_ops.band, host_ops.kind) == (0.05, None, "measured")
    assert find_term(MODEL.latency(walled), "total_s").value == pytest.approx(
        find_term(terms, "total_s").value + 0.05, rel=1e-12
    )


def test_latency_modeled_terms_use_the_record_census(record):
    terms = MODEL.latency(record)
    bandwidth = band_values(PROGRAMMING_BANDWIDTH_BYTES_PER_S)
    assert term_band(find_term(terms, "programming_s")) == (
        PAYLOAD_BYTES[0] / bandwidth[2],
        PAYLOAD_BYTES[1] / bandwidth[1],
        PAYLOAD_BYTES[2] / bandwidth[0],
    )
    assert term_band(find_term(terms, "core_init_s")) == pytest.approx(
        tuple(TOTAL_CORES * value for value in CORE_INIT_TIME), rel=1e-12
    )
    assert term_band(find_term(terms, "sync_s")) == pytest.approx(
        tuple(SYNC_COUNT * value for value in band_values(SYNC_BARRIER_S)), rel=1e-12
    )


def test_latency_total_adds_the_measured_and_modeled_terms_once(record):
    terms = MODEL.latency(record)
    modeled = [find_term(terms, name) for name in ("programming_s", "core_init_s", "sync_s")]
    total = find_term(terms, "total_s")
    for index, corner in enumerate(CORNERS):
        expected = COMPUTE_S + sum(term_band(term)[index] for term in modeled)
        assert term_band(total)[index] == pytest.approx(expected, rel=1e-12)


def test_latency_refuses_a_record_that_does_not_declare_the_noc_discipline(record):
    mangled = replace(
        record,
        timing=replace(
            record.timing,
            latency=replace(record.timing.latency, note="latency terms, no discipline"),
        ),
    )
    with pytest.raises(ValueError, match="no-double-count"):
        MODEL.latency(mangled)


def test_latency_requires_measured_compute_time(record):
    without_compute = replace(
        record,
        timing=replace(
            record.timing,
            latency=replace(record.timing.latency, compute_sim_time_s=None),
        ),
    )
    with pytest.raises(ValueError, match="compute_sim_time_s"):
        MODEL.latency(without_compute)


# ── area, throughput, report ────────────────────────────────────────────


def test_area_terms_are_the_utilization_census_and_its_derivations(record):
    terms = MODEL.area(record)
    crossbar = record.utilization.crossbar
    assert find_term(terms, "cores_used").value == float(crossbar.cores_allocated)
    assert find_term(terms, "cell_occupancy").value == crossbar.cell_occupancy
    assert find_term(terms, "cell_waste_fraction").value == 1.0 - crossbar.cell_occupancy
    assert find_term(terms, "cell_waste_fraction").kind == "derived"
    assert find_term(terms, "unused_area_cells").value == float(
        record.utilization.layout.unused_area_total
    )
    assert find_term(terms, "unusable_space_cells").value == float(crossbar.unusable_space)
    assert find_term(terms, "fragmentation_pct").value == pytest.approx(
        record.utilization.layout.fragmentation_pct
    )


def test_throughput_inverts_the_latency_band(record):
    total = find_term(MODEL.latency(record), "total_s")
    samples_per_s = find_term(MODEL.throughput(record), "samples_per_s")
    low, nominal, high = term_band(total)
    assert term_band(samples_per_s) == (1.0 / high, 1.0 / nominal, 1.0 / low)
    assert samples_per_s.value == 1.0 / nominal


def test_every_modeled_term_is_banded_and_monotone(report):
    modeled = [term for term in report.all_terms() if term.kind == "modeled"]
    assert len(modeled) >= 13
    for term in modeled:
        assert term.band is not None, term.name
        assert term.band.low <= term.band.nominal <= term.band.high, term.name
        assert term.band.low <= term.value <= term.band.high, term.name
        assert term.band.basis, term.name


def test_measured_terms_never_carry_a_band(report):
    measured = [term for term in report.all_terms() if term.kind == "measured"]
    assert len(measured) >= 7
    for term in measured:
        assert term.band is None, term.name


def test_every_term_states_a_unit_and_a_source(report):
    for term in report.all_terms():
        assert term.unit and term.source, term.name


def test_report_carries_the_records_no_double_count_note(record, report):
    assert record.timing.latency.note in report.notes


def test_report_json_round_trip_is_identity(report):
    payload = json.loads(json.dumps(report.to_dict()))
    assert DeploymentCostReport.from_dict(payload) == report


def test_report_rejects_a_foreign_format_version(report):
    payload = report.to_dict()
    payload["format_version"] = 2
    with pytest.raises(ValueError, match="format_version"):
        DeploymentCostReport.from_dict(payload)


# ── term invariants ─────────────────────────────────────────────────────


def test_a_measured_term_may_not_carry_a_band():
    with pytest.raises(ValueError, match="never presented"):
        CostTerm(
            name="x", unit="mJ", value=1.0,
            band=Band(0.5, 1.0, 2.0, basis="b"), kind="measured", source="s",
        )


def test_a_modeled_term_must_carry_a_band():
    with pytest.raises(ValueError, match="no band"):
        CostTerm(name="x", unit="mJ", value=1.0, band=None, kind="modeled", source="s")


def test_a_band_must_bracket_the_value_it_explains():
    with pytest.raises(ValueError, match="outside its band"):
        CostTerm(
            name="x", unit="mJ", value=9.0,
            band=Band(0.5, 1.0, 2.0, basis="b"), kind="modeled", source="s",
        )


def test_core_init_coefficients_are_derived_from_the_presets():
    assert band_values(CORE_INIT.energy_mj) == pytest.approx(CORE_INIT_ENERGY, rel=1e-12)
    assert band_values(CORE_INIT.time_s) == pytest.approx(CORE_INIT_TIME, rel=1e-12)
    assert "presets.py" in CORE_INIT.energy_mj.basis
