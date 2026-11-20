"""Wave 1 profile rows: every new row's evidence class and band, the unit fix, and the three
page-sourced profiles, each refusing by name what it does not declare."""

import pytest

from mimarsinan.chip_simulation.weight_reuse_cost_model import DEFAULT_COEFFICIENT_BAND
from mimarsinan.deployment_record.cost.absolute import ABSOLUTE_TERM_NAMES, price_absolute, price_write_terms
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import apply_overrides, get_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.load_census_fixtures import LOAD_TERMS

#: (profile, constant, evidence kind, note markers): the evidence-class mapping of plan §7.1.
NEW_ROWS = (
    ("generic_estimated_22nm", "endurance_cycles", "published", ()),
    ("generic_estimated_22nm", "e_write_per_cell", "derived", ()),
    ("generic_estimated_22nm", "write_verify_iterations", "derived", ()),
    ("generic_estimated_22nm", "e_noc_value_per_byte_hop", "estimated", ("Horowitz",)),
    ("odin", "endurance_cycles", "published", ()),
    # [U-C C-1] the per-technology bands: the resistive proxy (lee2011a) and the PCM review (gallo2020an)
    ("isaac_like", "endurance_cycles", "published", ("proxy", "same-technology transfer")),
    ("isaac_like", "write_verify_iterations", "published", ("same-technology transfer",)),
    ("isaac_like", "t_write_per_unit", "derived", ("same-technology transfer",)),
    ("isaac_like", "e_noc_value_per_byte_hop", "derived", ()),
    ("hermes_64", "e_mac", "published", ()),
    ("hermes_64", "t_cycle", "published", ()),
    ("hermes_64", "area_per_core_total", "published", ()),
    # [U-AREA] hermes_64's area_global_fixed (the derived die remainder) is dropped: its outside part is gamma
    ("hermes_64", "t_write_per_unit", "derived", ("per core iteration", "selected diagonal")),
    ("hermes_64", "write_verify_iterations", "published", ()),
    ("hermes_64", "endurance_cycles", "published", ("review", "Ge2Sb2Te5")),
    ("sram_cim_macro", "e_mac", "derived", ()),
    ("sram_cim_macro", "e_write_per_cell", "derived", ()),
    ("sram_cim_macro", "write_verify_iterations", "derived", ()),
    ("sram_cim_macro", "endurance_cycles", "published", ()),
    ("sram_cim_macro", "conductance_levels", "published", ()),
    ("sram_cim_macro", "t_cycle", "derived", ("16 32b outputs", "0.05us", "91MHz", "8/8/32", "DECLARED")),
    ("neurram", "e_mac", "published", ("TOPS/W 43 | 40 | 16 | 7",)),
    ("neurram", "t_cycle", "published", ()),
    ("neurram", "write_verify_iterations", "published", ()),
    ("neurram", "t_write_per_unit", "derived", ()),
    ("neurram", "endurance_cycles", "published", ("proxy", "HfOx")),
    ("neurram", "conductance_levels", "published", ()),
)


@pytest.mark.parametrize("profile, key, kind, markers", NEW_ROWS)
def test_each_new_row_carries_its_evidence_class_and_band(profile, key, kind, markers):
    value = get_platform_physics(profile).constants[key]
    assert value.evidence_kind == kind
    assert value.low <= value.nominal <= value.high
    for marker in markers:
        assert marker.lower() in (value.note + value.derivation).lower(), (profile, key, marker)
    if "SECONDARY" in markers:
        assert "secondary" in value.note.lower() and value.citation


@pytest.mark.parametrize("key", ["e_dma_per_byte", "e_readout_per_byte"])
def test_the_22nm_transfer_rows_equal_the_framework_dma_band_per_byte(key):
    band = get_platform_physics("generic_estimated_22nm").band(key)
    mj_to_j = 1e-3
    for corner in ("low", "nominal", "high"):
        framework = getattr(DEFAULT_COEFFICIENT_BAND, corner).e_dma_per_byte_mj * mj_to_j
        assert getattr(band, corner) == pytest.approx(framework, rel=1e-12)
    assert (band.low, band.nominal, band.high) == pytest.approx((31e-12, 160e-12, 320e-12))


def _census():
    return PassPeriodCensus(passes=(PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                                               written_cells_backing_store=100, written_cells_producer=0,
                                               payload_bytes=100, compute_steps=1),))


def _quantities(**values):
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _macro_with_store():
    """The macro with a test backing store (its external component) declared beside it."""
    rows = {"evidence_kind": "estimated", "note": "test store"}
    store = {"e_dma_per_byte": {"nominal": 100.0, "unit": "pJ", **rows},
             "t_program_per_byte": {"nominal": 0.1, "unit": "ns", **rows}}
    return apply_overrides(get_platform_physics("sram_cim_macro"), store)


class TestTheMacroCycleTime:
    """[consts] The macro's cycle time from its own paper (jiang2023a): one step of the value window is one
    evaluation of the logical core, 1024 outputs at 16 per clock cycle (p.2), 64 cycles at the 4-bit input and 128 at
    the 8-bit MAC of the paper's precision row (Fig. 8c), at the measured 91 MHz to 20 MHz clocks (p.3-4)."""

    def test_the_band_is_one_core_evaluation_between_the_measured_clocks(self):
        band = get_platform_physics("sram_cim_macro").band("t_cycle")
        assert band.low == pytest.approx(64 / 91e6, rel=1e-6)       # 4-bit inputs at 91 MHz: 703.3 ns
        assert band.nominal == pytest.approx(128 * 50e-9, rel=1e-12)  # 8-bit MACs at 20 MHz: 6.4 us
        assert band.high == pytest.approx(128 * 50e-9, rel=1e-12)

    def test_the_row_cites_the_macro_paper_by_page_and_quotes_it(self):
        value = get_platform_physics("sram_cim_macro").constants["t_cycle"]
        assert value.citation.startswith("jiang2023a") and "p.2" in value.citation and "Fig. 8c" in value.citation
        for quote in ("In total 16 32b outputs (SUM[31:0]) are computed each clock cycle for the macro.",
                      "clock period time 0.05us (20MHz)", "functional down to 20MHz at 0.58V VDDH and 91MHz at 1V VDDH",
                      "Precision(I/W/O) 8/8/32", "An 8bW*8bA MAC is defined as two ops"):
            assert quote in value.derivation, quote

    def test_with_its_store_the_macro_prices_latency_as_t_cycle_times_the_steps(self):
        macro = _macro_with_store()
        pricing = price_absolute(_quantities(cores_physical=1, synaptic_events=10, host_macs=0, latency_steps=7),
                                 macro)
        assert "e2e_latency_s" not in {r.name for r in pricing.refusals}
        latency = {t.name: t for t in pricing.terms}["e2e_latency_s"].band
        cycle = macro.band("t_cycle")
        assert (latency.low, latency.nominal, latency.high) == pytest.approx(
            (7 * cycle.low, 7 * cycle.nominal, 7 * cycle.high), rel=1e-12)

    def test_without_its_store_the_macro_still_prices_nothing(self):
        """The cycle time does not lift the external-component gate: no absolute term without the backing store."""
        pricing = price_absolute(_quantities(cores_physical=1, synaptic_events=10, host_macs=0, latency_steps=7),
                                 get_platform_physics("sram_cim_macro"))
        # [U-CR C-2, declared pin amendment] the gate's refusal, then each headline it suppresses by name, then
        # [U-ENERGY] each load figure of the one load law (refused by the law's own reason: no load census given)
        assert pricing.terms == () and [r.name for r in pricing.refusals] == [
            "absolute_pricing", *ABSOLUTE_TERM_NAMES, *LOAD_TERMS]


class TestThePageSourcedProfiles:
    @pytest.mark.parametrize("name", ["hermes_64", "sram_cim_macro", "neurram"])
    def test_each_profile_loads_with_its_validity(self, name):
        physics = get_platform_physics(name)
        assert physics.validity.measurement_kind == "silicon"
        assert physics.validity.geometry is not None

    def test_the_macro_refuses_its_chip_level_rows_by_name(self):
        """With its backing store declared (CL2: the profile's external component), the macro prices and still
        refuses every chip-level row it lacks by name; without the store it prices nothing
        (test_external_components). [consts] Its cycle time is declared, so a latency refusal names the absent
        latency census, never the constant."""
        pricing = price_absolute(_quantities(cores_physical=1, synaptic_events=10, host_macs=0), _macro_with_store())
        refused = {r.name: r.reason for r in pricing.refusals}
        # [U-AREA] the macro's area_rule block prices its chip area (R5 refused it: no area_per_core_total)
        area = {t.name: t for t in pricing.terms}["chip_area_mm2"]
        assert "area_rule T2_digital anchor (macro)" in area.source
        assert refused["e2e_latency_s"] == "t_cycle is declared but the latency_steps census is absent"

    def test_neurram_prices_its_write_energy_from_the_derived_band(self):
        """[U-C3, B3] NeuRRAM declares its write energy per pulse (a derived band, test_write_energy_b3_uc3): the
        write law prices it where it refused by name before; HERMES-64 still refuses (below)."""
        pricing = price_write_terms(_census(), get_platform_physics("neurram"), WritePolicy(batch_k=1))
        assert "energy_write_mj" in {t.name for t in pricing.terms}
        assert "energy_write_mj" not in {r.name for r in pricing.refusals}

    def test_hermes_prices_its_lifetime_and_refuses_its_write_energy(self):
        pricing = price_write_terms(_census(), get_platform_physics("hermes_64"), WritePolicy(batch_k=1),
                                    most_written=1)
        assert "lifetime_inferences" in {t.name for t in pricing.terms}
        assert "energy_write_mj" in {r.name for r in pricing.refusals}

    def test_hermes_prices_its_own_core_and_off_its_word_by_name(self):
        """[U-AREA] Its own point is the anchor's identity, the published die (R5: 64 x 1.392 + 54.912 = 144 mm2);
        8-bit weights over its 4-bit cells take two cells each, priced and named (R5 refused the area by name)."""
        values = _quantities(cores_physical=64, synaptic_events=10, host_macs=0, weight_bits=4)
        hermes = get_platform_physics("hermes_64")
        own = price_absolute(values, hermes, geometry=(CoreGeometry(256, 256, 64),))
        assert {t.name: t for t in own.terms}["chip_area_mm2"].value == pytest.approx(64 * 1.392 + 54.912)
        wide = price_absolute(_quantities(cores_physical=64, synaptic_events=10, host_macs=0, weight_bits=8),
                              hermes, geometry=(CoreGeometry(256, 256, 64),))
        area = {t.name: t for t in wide.terms}["chip_area_mm2"]
        assert "off word width: 8-bit weights over 4-bit cells take 2 cells" in area.source
