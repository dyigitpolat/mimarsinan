"""[U-C C-1 to C-3] Chip constants of the page-sourced profiles after the owner's fetch of 2026-10-27: each
non-volatile technology carries its own endurance band (RRAM from lee2011a, a device-class proxy; PCM from
gallo2020an, a review), each technology its pulses per write, NeuRRAM's MVM time and energy come from one operating
point of its Nature paper, and HERMES-64's write time is split into a per-core setup and a per-iteration step from its
published sequence. Every value states its source, page and class."""

import math

import pytest

from mimarsinan.deployment_record.cost.absolute import price_write_terms
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

#: profile -> (low, nominal, high, citation, markers): the per-technology endurance bands (fetch intake sec. 3.1).
ENDURANCE = {
    "neurram": (1e10, math.sqrt(1e10 * 1e11), 1e11, "lee2011a p.627", ("proxy", "HfOx", "Fig. 2c")),
    "isaac_like": (1e10, math.sqrt(1e10 * 1e11), 1e11, "lee2011a p.627", ("proxy", "same-technology transfer")),
    "hermes_64": (1e8, 1e10, 1e12, "gallo2020an p.4", ("review", "10^8", "10^12", "Ge2Sb2Te5")),
}
#: profile -> (low, nominal, high, evidence kind, a marker of its source): pulses per write.
PULSES = {
    "neurram": (8.52, 8.52, 8.52, "published", "8.52 set/reset pulses"),
    "isaac_like": (8.52, 8.52, 8.52, "published", "same-technology transfer"),
    "hermes_64": (3.0, 7.4772, 32.0, "derived", "first RESET"),
    "generic_estimated_22nm": (1.0, 1.0, 1.0, "derived", "ielmini2018inmemory"),
    "odin": (1.0, 1.0, 1.0, "derived", "ielmini2018inmemory"),
    "sram_cim_macro": (1.0, 1.0, 1.0, "derived", "ielmini2018inmemory"),
}


def _row(profile, key):
    return get_platform_physics(profile).constants[key]


class TestEnduranceBands:
    @pytest.mark.parametrize("profile", sorted(ENDURANCE))
    def test_each_non_volatile_technology_has_its_own_band(self, profile):
        low, nominal, high, citation, markers = ENDURANCE[profile]
        row = _row(profile, "endurance_cycles")
        assert (row.low, row.nominal, row.high) == pytest.approx((low, nominal, high), rel=1e-4)
        assert row.evidence_kind == "published" and row.citation == citation
        text = row.note + row.derivation
        for marker in markers:
            assert marker in text, (profile, marker)
        assert "sridharan2023xformer" not in row.citation and "SECONDARY" not in row.note

    def test_no_profile_keeps_the_shared_secondary_band(self):
        for name in available_profiles():
            physics = get_platform_physics(name)
            if physics.has("endurance_cycles"):
                assert physics.constants["endurance_cycles"].citation != "sridharan2023xformer p.1", name

    def test_the_sram_rows_are_unchanged(self):
        for name in ("generic_estimated_22nm", "odin", "sram_cim_macro"):
            row = _row(name, "endurance_cycles")
            assert (row.low, row.nominal, row.high, row.citation) == (1e16, 1e16, 1e16, "ielmini2018inmemory p.7")


class TestPulsesPerWrite:
    @pytest.mark.parametrize("profile", sorted(PULSES))
    def test_every_technology_with_an_endurance_declares_its_pulses(self, profile):
        low, nominal, high, kind, marker = PULSES[profile]
        row = _row(profile, "pulses_per_write")
        assert (row.low, row.nominal, row.high) == pytest.approx((low, nominal, high))
        assert row.evidence_kind == kind and row.unit == "pulses"
        assert marker in row.note + row.derivation + row.citation, profile

    def test_every_profile_with_an_endurance_declares_pulses(self):
        for name in available_profiles():
            physics = get_platform_physics(name)
            assert physics.has("pulses_per_write") == physics.has("endurance_cycles"), name

    def test_neurram_cites_the_nature_methods_page(self):
        row = _row("neurram", "pulses_per_write")
        assert row.citation == "wan2022a Methods, PDF p.10"
        assert "On average each cell requires 8.52 set/reset pulses." in row.note

    def test_hermes_64_counts_reset_set_and_one_pulse_per_iteration(self):
        """2 + write_verify_iterations corner by corner, from the published sequence (p.690)."""
        pulses, iterations = _row("hermes_64", "pulses_per_write"), _row("hermes_64", "write_verify_iterations")
        assert (pulses.low, pulses.nominal, pulses.high) == pytest.approx(
            (2 + iterations.low, 2 + iterations.nominal, 2 + iterations.high))
        assert "p.690" in pulses.citation


class TestNeuRRAMOperatingPoint:
    """C-2: e_mac and t_cycle from one row of wan2022a Extended Data Table 1 (4-bit inputs, 4-bit weights, 6-bit
    outputs), with the energy-delay product the table prints as the check."""

    def test_t_cycle_is_the_latency_at_the_e_mac_point(self):
        row = _row("neurram", "t_cycle")
        assert (row.low, row.nominal, row.high, row.unit) == (3.9, 3.9, 3.9, "us")
        assert row.citation == "wan2022a Extended Data Table 1, PDF p.28" and row.evidence_kind == "published"
        assert "Latency of a 256x256 MVM" in row.note and "32.0" in row.note

    def test_e_mac_is_primary_at_the_same_point(self):
        row = _row("neurram", "e_mac")
        assert row.nominal == 125.0 and row.unit == "fJ" and row.evidence_kind == "published"
        assert row.citation == "wan2022a Extended Data Table 1, PDF p.28"
        assert "SECONDARY" not in row.note and "TOPS/W 43 | 40 | 16 | 7" in row.note

    def test_the_energy_delay_product_closes_on_the_printed_value(self):
        e_mac, t_cycle = (get_platform_physics("neurram").band(k).nominal for k in ("e_mac", "t_cycle"))
        edp = 256 * 256 * e_mac * t_cycle  # J s per 256 x 256 MVM
        assert edp == pytest.approx(32.0e-15, rel=0.01, abs=0)

    def test_the_write_rows_cite_the_nature_methods_page(self):
        for key in ("write_verify_iterations", "t_write_per_unit"):
            assert "wan2022a Methods, PDF p.10" in _row("neurram", key).citation, key


class TestHermesWriteTime:
    """C-3: a per-core setup (RESET all four devices, SET the weight's device, 32 in parallel) and a per-iteration
    step (one 125 ns pulse per unit cell, 32 in parallel, then one verify round of the 256 diagonals)."""

    def test_the_setup_per_core(self):
        row = _row("hermes_64", "t_write_setup_per_event")
        assert (row.low, row.nominal, row.high, row.unit) == pytest.approx((1.536, 1.536, 1.6384, "ms"))
        assert row.evidence_kind == "derived"
        for fact in ("first RESET", "32 SLs", "125 ns", "250 ns", "p.689", "p.690"):
            assert fact in row.derivation, fact

    def test_the_step_per_iteration(self):
        row = _row("hermes_64", "t_write_per_unit")
        assert (row.low, row.nominal, row.high) == pytest.approx((0.452608, 0.649216, 1.042432))
        assert row.unit == "ms" and row.evidence_kind == "derived"
        for fact in ("125 ns width", "512 ns width and 256 ns precharge", "selected diagonal", "p.690"):
            assert fact in row.derivation, fact
        assert "LOWER bound" not in row.note

    def test_one_core_write_is_setup_plus_steps_times_iterations(self):
        """At granularity core: 1.99 ms at one iteration, 5.09 ms nominal, 32.9 ms at the 30-iteration maximum."""
        census = PassPeriodCensus(passes=tuple(
            PassCensus(position=i, segment_index=0, resident=False, cores=1, resident_cores=0,
                       written_cells_backing_store=65536, written_cells_producer=0, payload_bytes=32768,
                       compute_steps=1) for i in (0, 1)))
        pricing = price_write_terms(census, get_platform_physics("hermes_64"),
                                    WritePolicy(batch_k=1, write_granularity="core"), most_written=2)
        exposed = {t.name: t for t in pricing.terms}["latency_write_exposed_s"].band
        per_core = [v / 2 * 1e3 for v in (exposed.low, exposed.nominal, exposed.high)]
        assert per_core == pytest.approx([0.452608 * 1 + 1.536, 0.649216 * 5.4772 + 1.536,
                                          1.042432 * 30 + 1.6384])
