"""The quantity catalog: the closed, dimensioned list of priceable multiplicands."""

import pytest

from mimarsinan.deployment_record.units import DIMENSIONS, unit_for
from mimarsinan.deployment_record.quantities.spec import (
    PROVENANCE_KINDS,
    QUANTITY_SPECS,
    Quantities,
    QuantityValue,
    quantity_spec,
)


def test_every_spec_key_matches_its_mapping_key():
    for key, spec in QUANTITY_SPECS.items():
        assert spec.key == key


def test_every_spec_declares_a_known_dimension_and_matching_unit():
    for spec in QUANTITY_SPECS.values():
        assert spec.dimension in DIMENSIONS, spec.key
        assert unit_for(spec.unit).dimension == spec.dimension, spec.key


def test_every_spec_documents_itself():
    for spec in QUANTITY_SPECS.values():
        assert len(spec.doc.strip()) >= 20, spec.key


def test_the_two_mac_censuses_are_distinct_quantities():
    """As-mapped MAC sites (energy multiplicand — replicas really fire) vs the logical
    forward-MAC census (the host/chip split) must never be conflated."""
    assert "macs" in QUANTITY_SPECS
    assert "total_macs" in QUANTITY_SPECS
    assert "as-mapped" in QUANTITY_SPECS["macs"].doc.lower()
    assert "logical" in QUANTITY_SPECS["total_macs"].doc.lower()


def test_unknown_quantity_raises_naming_the_catalog():
    with pytest.raises(KeyError, match="q_made_up"):
        quantity_spec("q_made_up")


def test_quantity_value_rejects_unknown_provenance():
    assert PROVENANCE_KINDS == frozenset({"measured", "static", "modeled"})
    with pytest.raises(ValueError, match="provenance"):
        QuantityValue(value=1.0, provenance="vibes")


def test_quantities_reject_keys_outside_the_catalog():
    with pytest.raises(KeyError, match="q_made_up"):
        Quantities({"q_made_up": QuantityValue(1.0, "static")})


def test_quantities_absence_is_meaningful_and_never_defaults():
    quantities = Quantities({"pass_count": QuantityValue(2.0, "measured")})
    assert quantities.has("pass_count")
    assert quantities.get("pass_count").value == 2.0
    assert not quantities.has("sync_count")
    with pytest.raises(KeyError, match="sync_count"):
        quantities.get("sync_count")
    assert quantities.missing(("pass_count", "sync_count", "tiles")) == (
        "sync_count",
        "tiles",
    )


def test_quantities_iterate_in_catalog_order():
    quantities = Quantities({
        "sync_count": QuantityValue(1.0, "measured"),
        "pass_count": QuantityValue(2.0, "measured"),
    })
    catalog_order = [k for k in QUANTITY_SPECS if quantities.has(k)]
    assert list(quantities.keys()) == catalog_order


def test_the_catalog_covers_the_pricing_surfaces():
    """Program/dynamics/host groups each need their multiplicands present."""
    expected = {
        # structural
        "cells_used", "cells_physical", "cores_allocated", "neurons_used",
        "neurons_physical", "axons_used", "axons_physical", "tiles", "weight_bits",
        # program
        "pass_count", "sync_count", "reprogram_passes", "reprogrammed_bytes",
        "connectivity_entries", "segment_cores", "reprogrammed_cores",
        # dynamics
        "timesteps", "latency_steps", "total_spikes", "boundary_events",
        "synaptic_events", "noc_total_packets", "noc_intra_tile_packets",
        "noc_inter_tile_packets", "noc_total_hops",
        # host / split
        "host_ops_s", "total_params", "onchip_params", "host_params",
        "total_macs", "onchip_macs", "host_macs", "macs",
        # periphery producers pending (C6 conversion models)
        "adc_conversions", "adc_count",
    }
    assert expected <= set(QUANTITY_SPECS)
