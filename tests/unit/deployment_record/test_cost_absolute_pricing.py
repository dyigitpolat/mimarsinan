"""The analytical pricer: term = constant band × quantity, refusals written, no double count."""

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import (
    get_platform_physics,
    profile_from_dict,
)
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

from unit.deployment_record.record_fixtures import make_full_record

_TRUENORTH = get_platform_physics("truenorth")


def _quantities(**values):
    return Quantities({
        key: QuantityValue(float(value), "static") for key, value in values.items()
    })


def _by_name(pricing):
    return {term.name: term for term in pricing.terms}


def _reasons(pricing):
    return {refusal.name: refusal.reason for refusal in pricing.refusals}


class TestArea:
    def test_truenorth_area_is_cores_times_footprint_plus_global(self):
        pricing = price_absolute(_quantities(cores_physical=20, host_macs=0), _TRUENORTH)
        term = _by_name(pricing)["chip_area_mm2"]
        assert term.value == pytest.approx(20 * 0.0936 + 46.6)
        assert term.unit == "mm^2"
        assert term.kind == "modeled"
        assert term.band is not None

    def test_the_area_aggregate_supersedes_the_decomposition(self):
        """TrueNorth declares BOTH the measured footprint and per-cell figures; only
        the aggregate may be charged, or every component counts twice."""
        pricing = price_absolute(
            _quantities(cores_physical=2, cells_physical=131072, neurons_physical=512,
                        host_macs=0),
            _TRUENORTH,
        )
        term = _by_name(pricing)["chip_area_mm2"]
        assert term.value == pytest.approx(2 * 0.0936 + 46.6)

    def test_area_evidence_rides_in_the_basis(self):
        pricing = price_absolute(_quantities(cores_physical=1, host_macs=0), _TRUENORTH)
        assert "akopyan2015truenorth" in _by_name(pricing)["chip_area_mm2"].source

    def test_area_refused_without_a_compute_area_source(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T",
            "description_file": "t.md",
            "validity": {"measurement_kind": "silicon"},
            "constants": {"t_cycle": {"nominal": 1.0, "unit": "ms",
                                      "evidence_kind": "estimated", "note": "n"}},
        })
        pricing = price_absolute(_quantities(cores_physical=4, host_macs=0), physics)
        assert "chip_area_mm2" not in _by_name(pricing)
        assert "area_per_core_total" in _reasons(pricing)["chip_area_mm2"]

    def test_area_refused_without_the_core_census(self):
        pricing = price_absolute(_quantities(host_macs=0), _TRUENORTH)
        assert "cores_physical" in _reasons(pricing)["chip_area_mm2"]


class TestEnergy:
    def test_truenorth_energy_needs_the_event_census(self):
        """The sealed record has no synaptic-event census yet: the aggregate cannot
        multiply, and the headline refuses BY NAME instead of guessing."""
        pricing = price_absolute(from_record(make_full_record()), _TRUENORTH)
        assert "energy_per_inference_mj" not in _by_name(pricing)
        assert "synaptic_events" in _reasons(pricing)["energy_per_inference_mj"]

    def test_truenorth_energy_prices_the_aggregate_over_events(self):
        pricing = price_absolute(
            _quantities(synaptic_events=1e6, cores_physical=20, latency_steps=32,
                        host_macs=0),
            _TRUENORTH,
        )
        term = _by_name(pricing)["energy_per_inference_mj"]
        # 26 pJ x 1e6 events = 26 uJ = 0.026 mJ; static power is superseded by the
        # aggregate, and TrueNorth declares no sync energy (noted, not added).
        assert term.value == pytest.approx(26e-12 * 1e6 * 1e3)
        assert term.unit == "mJ"
        assert "e_synaptic_event_total" in term.source

    def test_unpriced_components_are_named_on_the_headline(self):
        pricing = price_absolute(
            _quantities(synaptic_events=1e6, sync_count=3, host_macs=0),
            _TRUENORTH,
        )
        term = _by_name(pricing)["energy_per_inference_mj"]
        assert "e_sync_barrier" in term.source, "sync work exists but is unpriced: say so"

    def test_host_work_without_host_constants_refuses_the_energy_headline(self):
        """The §8b rig: host MACs exist, no host pricing declared — a headline that
        silently omitted them would make subsume look free."""
        pricing = price_absolute(
            _quantities(synaptic_events=1e6, host_macs=5000, host_ops_s=0.5),
            _TRUENORTH,
        )
        assert "energy_per_inference_mj" not in _by_name(pricing)
        assert "p_host" in _reasons(pricing)["energy_per_inference_mj"]

    def test_known_zero_host_work_needs_no_host_constants(self):
        pricing = price_absolute(
            _quantities(synaptic_events=1e6, host_macs=0), _TRUENORTH
        )
        assert "energy_per_inference_mj" in _by_name(pricing)

    def test_host_energy_prices_the_measured_wall(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T",
            "description_file": "t.md",
            "validity": {"measurement_kind": "mixed"},
            "constants": {
                "e_synaptic_event_total": {"nominal": 10.0, "unit": "pJ",
                                           "evidence_kind": "estimated", "note": "n"},
                "p_host": {"nominal": 40.0, "unit": "W",
                           "evidence_kind": "estimated", "note": "n"},
                "host_compute_rate": {"nominal": 2.0, "unit": "fraction",
                                      "evidence_kind": "estimated", "note": "n"},
            },
        })
        pricing = price_absolute(
            _quantities(synaptic_events=1e6, host_macs=5000, host_ops_s=0.5), physics
        )
        by_name = _by_name(pricing)
        # Host wall on the deployment host = 0.5 s / 2.0; energy = 40 W x 0.25 s.
        assert by_name["energy_host_mj"].value == pytest.approx(40 * 0.25 * 1e3)
        assert by_name["energy_per_inference_mj"].value == pytest.approx(
            10e-12 * 1e6 * 1e3 + 40 * 0.25 * 1e3
        )


class TestLatency:
    def test_e2e_is_t_cycle_times_steps_plus_host(self):
        pricing = price_absolute(
            _quantities(latency_steps=64, host_macs=0), _TRUENORTH
        )
        term = _by_name(pricing)["e2e_latency_s"]
        assert term.value == pytest.approx(64 * 1e-3)
        assert term.unit == "s"

    def test_the_band_carries_the_demonstrated_fast_corner(self):
        """TrueNorth's t_cycle band is one-sided (47.6 us demonstrated, 1 ms nominal):
        the priced latency band must carry it rather than flattening it."""
        pricing = price_absolute(_quantities(latency_steps=64, host_macs=0), _TRUENORTH)
        band = _by_name(pricing)["e2e_latency_s"].band
        assert band is not None
        assert band.low == pytest.approx(64 * 47.6e-6)
        assert band.high == pytest.approx(64 * 1e-3)

    def test_programming_is_an_overhead_term_not_the_steady_state_headline(self):
        pricing = price_absolute(
            _quantities(latency_steps=64, reprogrammed_bytes=1000,
                        connectivity_entries=24, host_macs=0),
            _TRUENORTH,
        )
        by_name = _by_name(pricing)
        # 800 ns/B over (1000 payload + 24 entries x 4 B) = 1096 B.
        assert by_name["latency_programming_s"].value == pytest.approx(1096 * 800e-9)
        assert by_name["e2e_latency_s"].value == pytest.approx(64 * 1e-3), (
            "steady-state headline: resident weights, programming amortized separately"
        )

    def test_hop_time_is_never_added_to_e2e(self):
        """Execution is timestep-synchronous: hop time is already INSIDE the timestep
        t_cycle prices. Adding t_hop x hops would double-count it — the same
        discipline the measured sim_time_s carries."""
        physics = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T",
            "description_file": "t.md",
            "validity": {"measurement_kind": "mixed"},
            "constants": {
                "t_cycle": {"nominal": 1.0, "unit": "ms",
                            "evidence_kind": "estimated", "note": "n"},
                "t_hop": {"nominal": 5.0, "unit": "ns",
                          "evidence_kind": "estimated", "note": "n"},
            },
        })
        pricing = price_absolute(
            _quantities(latency_steps=10, noc_total_hops=1e6, host_macs=0), physics
        )
        term = _by_name(pricing)["e2e_latency_s"]
        assert term.value == pytest.approx(10 * 1e-3), (
            "t_hop x 1e6 hops would add 5 ms; hop time is already in the timestep"
        )
        assert "t_hop" not in term.source

    def test_latency_refused_without_the_time_converter(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T",
            "description_file": "t.md",
            "validity": {"measurement_kind": "silicon"},
            "constants": {"e_synaptic_event_total": {
                "nominal": 26.0, "unit": "pJ", "evidence_kind": "estimated",
                "note": "n"}},
        })
        pricing = price_absolute(_quantities(latency_steps=64, host_macs=0), physics)
        assert "t_cycle" in _reasons(pricing)["e2e_latency_s"]

    def test_host_time_models_from_the_declared_rate_when_no_wall_exists(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T",
            "description_file": "t.md",
            "validity": {"measurement_kind": "mixed"},
            "constants": {
                "t_cycle": {"nominal": 1.0, "unit": "us",
                            "evidence_kind": "estimated", "note": "n"},
                "p_host": {"nominal": 40.0, "unit": "W",
                           "evidence_kind": "estimated", "note": "n"},
                "host_compute_rate": {"nominal": 1.0, "unit": "fraction",
                                      "evidence_kind": "estimated", "note": "n"},
                "host_macs_per_s": {"nominal": 2.0, "unit": "G/s",
                                    "evidence_kind": "estimated", "note": "n"},
            },
        })
        pricing = price_absolute(
            _quantities(latency_steps=32, host_macs=4e9), physics
        )
        by_name = _by_name(pricing)
        assert by_name["latency_host_s"].value == pytest.approx(2.0)
        assert by_name["e2e_latency_s"].value == pytest.approx(32e-6 + 2.0)
        assert by_name["latency_host_s"].kind == "modeled"


class TestThroughputAndStatic:
    def test_throughput_inverts_the_steady_state_latency(self):
        pricing = price_absolute(_quantities(latency_steps=64, host_macs=0), _TRUENORTH)
        by_name = _by_name(pricing)
        assert by_name["throughput_inferences_s"].value == pytest.approx(1.0 / 0.064)
        band = by_name["throughput_inferences_s"].band
        assert band is not None
        assert band.high == pytest.approx(1.0 / (64 * 47.6e-6)), "corners flip on inversion"

    def test_static_energy_multiplies_the_priced_latency_when_not_superseded(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T",
            "description_file": "t.md",
            "validity": {"measurement_kind": "mixed"},
            "constants": {
                "e_mac": {"nominal": 1.0, "unit": "pJ",
                          "evidence_kind": "estimated", "note": "n"},
                "t_cycle": {"nominal": 1.0, "unit": "ms",
                            "evidence_kind": "estimated", "note": "n"},
                "p_static_per_core": {"nominal": 10.0, "unit": "uW",
                                      "evidence_kind": "estimated", "note": "n"},
            },
        })
        pricing = price_absolute(
            _quantities(synaptic_events=1e6, latency_steps=10, cores_physical=100,
                        host_macs=0),
            physics,
        )
        by_name = _by_name(pricing)
        # 10 uW x 100 cores x 10 ms = 10 uJ = 0.01 mJ.
        assert by_name["energy_static_mj"].value == pytest.approx(0.01)
        assert by_name["energy_per_inference_mj"].value == pytest.approx(
            1e-12 * 1e6 * 1e3 + 0.01
        )


class TestDiscipline:
    def test_every_priced_term_is_modeled_with_a_band(self):
        pricing = price_absolute(
            _quantities(cores_physical=20, latency_steps=64, synaptic_events=1e6,
                        host_macs=0),
            _TRUENORTH,
        )
        assert pricing.terms, "the profile prices something"
        for term in pricing.terms:
            assert term.kind == "modeled", term.name
            assert term.band is not None, term.name

    def test_refusals_never_overlap_terms(self):
        pricing = price_absolute(from_record(make_full_record()), _TRUENORTH)
        produced = {term.name for term in pricing.terms}
        refused = set(_reasons(pricing))
        assert not produced & refused


class TestTheConversionModel:
    """An analog target prices conversion because its DATAFLOW says how much of it
    there is — a count no deployment record could have carried."""

    def _analog(self, **over):
        constants = {
            "e_adc_conversion": {"nominal": 2.0, "unit": "pJ",
                                 "evidence_kind": "estimated", "note": "n"},
            "area_per_adc": {"nominal": 1000.0, "unit": "um^2",
                             "evidence_kind": "estimated", "note": "n"},
            "area_per_cell": {"nominal": 0.1, "unit": "um^2",
                              "evidence_kind": "estimated", "note": "n"},
            "e_mac": {"nominal": 0.1, "unit": "pJ",
                      "evidence_kind": "estimated", "note": "n"},
            "t_cycle": {"nominal": 1.0, "unit": "us",
                        "evidence_kind": "estimated", "note": "n"},
        }
        constants.update(over)
        return profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T",
            "description_file": "t.md",
            "validity": {"measurement_kind": "projection"},
            "conversion_model": {
                "model": "bit_sliced_crossbar", "array_rows": 128,
                "array_cols": 128, "adc_sharing_factor": 8, "input_bits": 16,
                "cells_per_weight": 1, "dac_bits": 1,
            },
            "constants": constants,
        })

    def test_conversion_energy_is_priced_from_the_derived_count(self):
        pricing = price_absolute(
            _quantities(synaptic_events=1e5, macs=128 * 1000, host_macs=0), self._analog()
        )
        term = _by_name(pricing)["energy_per_inference_mj"]
        # 1000 integrations x 16 slices x 2 pJ = 32 uJ, on top of the MAC energy.
        assert term.value == pytest.approx(1e5 * 0.1e-12 * 1e3 + 1000 * 16 * 2e-12 * 1e3)

    def test_converter_area_is_priced_from_the_derived_count(self):
        pricing = price_absolute(
            _quantities(cells_physical=128 * 128 * 4, host_macs=0), self._analog()
        )
        area = _by_name(pricing)["chip_area_mm2"]
        # 4 arrays x 16 ADCs x 1000 um^2, plus the cells' own area.
        assert area.value == pytest.approx(
            (4 * 16 * 1000.0 + 128 * 128 * 4 * 0.1) * 1e-6
        )

    def test_a_digital_target_prices_no_conversion(self):
        pricing = price_absolute(
            _quantities(synaptic_events=1e6, macs=1e6, host_macs=0), _TRUENORTH
        )
        term = _by_name(pricing)["energy_per_inference_mj"]
        assert term.value == pytest.approx(26e-12 * 1e6 * 1e3), (
            "the aggregate already contains everything; a zero conversion count adds "
            "nothing and must not be mistaken for a missing one"
        )

    def test_a_view_quantity_always_wins_over_a_modeled_one(self):
        """If a record ever DOES seal a conversion census, the measurement must beat
        the model rather than the model silently overwriting it."""
        pricing = price_absolute(
            _quantities(macs=128 * 1000, adc_conversions=7.0, synaptic_events=0,
                        host_macs=0),
            self._analog(),
        )
        term = _by_name(pricing)["energy_per_inference_mj"]
        assert term.value == pytest.approx(7.0 * 2e-12 * 1e3)
