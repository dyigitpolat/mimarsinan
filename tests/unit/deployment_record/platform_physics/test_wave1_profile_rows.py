"""Wave 1 profile rows: every new row's evidence class and band, the unit fix, and the three
page-sourced profiles, each refusing by name what it does not declare."""

import pytest

from mimarsinan.chip_simulation.weight_reuse_cost_model import DEFAULT_COEFFICIENT_BAND
from mimarsinan.deployment_record.cost.absolute import price_absolute, price_write_terms
from mimarsinan.deployment_record.cost.absolute.validity import CoreGeometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

#: (profile, constant, evidence kind, note markers): the evidence-class mapping of plan §7.1.
NEW_ROWS = (
    ("generic_estimated_22nm", "endurance_cycles", "published", ()),
    ("generic_estimated_22nm", "e_write_per_cell", "derived", ()),
    ("generic_estimated_22nm", "write_verify_iterations", "derived", ()),
    ("generic_estimated_22nm", "e_noc_value_per_byte_hop", "estimated", ("Horowitz",)),
    ("odin", "endurance_cycles", "published", ()),
    ("isaac_like", "endurance_cycles", "published", ("SECONDARY", "lee2011a")),
    ("isaac_like", "write_verify_iterations", "published", ("same-technology transfer",)),
    ("isaac_like", "t_write_per_unit", "derived", ("same-technology transfer",)),
    ("isaac_like", "e_noc_value_per_byte_hop", "derived", ()),
    ("hermes_64", "e_mac", "published", ()),
    ("hermes_64", "t_cycle", "published", ()),
    ("hermes_64", "area_per_core_total", "published", ()),
    ("hermes_64", "area_global_fixed", "derived", ()),
    ("hermes_64", "t_write_per_unit", "derived", ("LOWER bound",)),
    ("hermes_64", "write_verify_iterations", "published", ()),
    ("hermes_64", "endurance_cycles", "published", ("SECONDARY",)),
    ("sram_cim_macro", "e_mac", "derived", ()),
    ("sram_cim_macro", "e_write_per_cell", "derived", ()),
    ("sram_cim_macro", "write_verify_iterations", "derived", ()),
    ("sram_cim_macro", "endurance_cycles", "published", ()),
    ("sram_cim_macro", "conductance_levels", "published", ()),
    ("neurram", "e_mac", "published", ("SECONDARY",)),
    ("neurram", "t_cycle", "published", ()),
    ("neurram", "write_verify_iterations", "published", ()),
    ("neurram", "t_write_per_unit", "derived", ()),
    ("neurram", "endurance_cycles", "published", ("SECONDARY",)),
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


class TestThePageSourcedProfiles:
    @pytest.mark.parametrize("name", ["hermes_64", "sram_cim_macro", "neurram"])
    def test_each_profile_loads_with_its_validity(self, name):
        physics = get_platform_physics(name)
        assert physics.validity.measurement_kind == "silicon"
        assert physics.validity.geometry is not None

    def test_the_macro_refuses_its_chip_level_rows_by_name(self):
        pricing = price_absolute(_quantities(cores_physical=1, synaptic_events=10, host_macs=0),
                                 get_platform_physics("sram_cim_macro"))
        refused = {r.name: r.reason for r in pricing.refusals}
        assert "area_per_core_total" in refused["chip_area_mm2"]
        assert "t_cycle" in refused["e2e_latency_s"]

    def test_neurram_refuses_its_write_energy_by_name(self):
        pricing = price_write_terms(_census(), get_platform_physics("neurram"), WritePolicy(batch_k=1))
        assert "e_write_per_cell" in {r.name: r.reason for r in pricing.refusals}["energy_write_mj"]

    def test_hermes_prices_its_lifetime_and_refuses_its_write_energy(self):
        pricing = price_write_terms(_census(), get_platform_physics("hermes_64"), WritePolicy(batch_k=1),
                                    most_written=1)
        assert "lifetime_inferences" in {t.name for t in pricing.terms}
        assert "energy_write_mj" in {r.name for r in pricing.refusals}

    def test_hermes_prices_area_only_at_its_own_core(self):
        values = _quantities(cores_physical=64, synaptic_events=10, host_macs=0, weight_bits=4)
        hermes = get_platform_physics("hermes_64")
        own = price_absolute(values, hermes, geometry=(CoreGeometry(256, 256, 64),))
        assert {t.name: t for t in own.terms}["chip_area_mm2"].value == pytest.approx(64 * 1.392 + 54.912)
        wide = price_absolute(_quantities(cores_physical=64, synaptic_events=10, host_macs=0, weight_bits=8),
                              hermes, geometry=(CoreGeometry(256, 256, 64),))
        assert "chip_area_mm2" in {r.name for r in wide.refusals}
