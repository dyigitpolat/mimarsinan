"""[E-5, W34] Converter energy, time and area over resolution: the pricer reads ``readout_law.adc_bits``.

A physics profile may state its converter as a table over resolution (rows of ``bits`` with ``e_adc_conversion``,
``t_adc_conversion`` and ``area_per_adc``, each with its evidence) and, for the energy, a cited scaling law anchored at
one of its rows (``energy_law``: E(b) = E(a) x (k1 b + k2 4^b) / (k1 a + k2 4^a), the form of houshmand2023benchmarking
Eq. 8, valid over its declared bit range). ``price_absolute`` takes the run's ``readout_law.adc_bits``
(``readout_bits``): the converter constants at that resolution price the converter terms; a resolution the table and
the law do not state refuses by name every headline whose converter work exists (energy for conversions, area for
converters). A profile that states no converter, or a run that declares no adc_bits, prices exactly as before.
Before E-5 the pricer never read adc_bits, so a 4-bit point and a 32-bit point of one bundle carried the same
converter energy and area (W34 plan, the named capability gap).

The new interface is imported inside each test so that each test fails on its own on a tree without it.
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from unit.deployment_record.g3_fixtures import quantities

E4, E8 = 0.5, 1.5625  # pJ per conversion in the hand table rows
A8 = 1200.0  # um^2 per converter at 8 bits


def _row(bits, e=None, t=None, a=None):
    row = {"bits": bits}
    for key, value, unit in (("e_adc_conversion", e, "pJ"), ("t_adc_conversion", t, "ns"), ("area_per_adc", a, "um^2")):
        if value is not None:
            row[key] = {"nominal": value, "unit": unit, "evidence_kind": "published", "citation": "hand table"}
    return row


def _profile(converter):
    """isaac_like's physics with a hand converter declaration (the shipped constants otherwise)."""
    from mimarsinan.deployment_record.platform_physics.converter import with_converter
    return with_converter(get_platform_physics("isaac_like"), converter)


def _q(**extra):
    return quantities(synaptic_events=1000, timesteps=1, adc_conversions=100, adc_count=4, cells_physical=1000,
                      **extra)


def _term(pricing, name):
    return next((t.band for t in pricing.terms if t.name == name), None)


def _refusal(pricing, name):
    return next((r.reason for r in pricing.refusals if r.name == name), None)


def _adc_energy(pricing):
    """The converter's share of the dynamic energy, read off the term's sources (``e_adc_conversion x``)."""
    return _term(pricing, "energy_dynamic_mj")


def test_a_profile_without_a_converter_table_prices_as_before_at_any_adc_bits():
    physics = get_platform_physics("generic_estimated_22nm")  # a digital array: converter constants 0, no table
    assert physics.converter is None
    assert price_absolute(_q(), physics, readout_bits=4) == price_absolute(_q(), physics)


def test_no_declared_adc_bits_prices_the_flat_constants():
    physics = _profile({"rows": [_row(8, e=E8, a=A8), _row(4, e=E4)]})
    flat = price_absolute(_q(), physics)
    assert flat == price_absolute(_q(), physics, readout_bits=None)
    assert _refusal(flat, "energy_per_inference_mj") is None or "adc_bits" not in _refusal(flat,
                                                                                         "energy_per_inference_mj")


def test_the_row_at_the_declared_resolution_prices_the_converter():
    physics = _profile({"rows": [_row(8, e=E8, a=A8), _row(4, e=E4, a=600.0)]})
    at8 = _adc_energy(price_absolute(_q(), physics, readout_bits=8))
    at4 = _adc_energy(price_absolute(_q(), physics, readout_bits=4))
    # 100 conversions: the difference of the two rows' energies, in mJ
    assert at8.nominal - at4.nominal == pytest.approx(100 * (E8 - E4) * 1e-12 * 1e3, rel=1e-9)
    a8 = _term(price_absolute(_q(), physics, readout_bits=8), "chip_area_mm2")
    a4 = _term(price_absolute(_q(), physics, readout_bits=4), "chip_area_mm2")
    assert a8.nominal - a4.nominal == pytest.approx(4 * (A8 - 600.0) * 1e-6, rel=1e-9)


def test_a_resolution_outside_the_table_refuses_the_converter_headlines_by_name():
    physics = _profile({"rows": [_row(8, e=E8, a=A8)]})
    pricing = price_absolute(_q(), physics, readout_bits=6)
    for name in ("energy_per_inference_mj", "energy_dynamic_mj", "chip_area_mm2"):
        reason = _refusal(pricing, name)
        assert reason is not None and "readout_law.adc_bits 6" in reason and "(8)" in reason, name
    assert _term(pricing, "energy_dynamic_mj") is None and _term(pricing, "chip_area_mm2") is None


def test_no_converter_work_refuses_nothing_outside_the_table():
    physics = _profile({"rows": [_row(8, e=E8, a=A8)]})
    pricing = price_absolute(quantities(synaptic_events=1000, timesteps=1, adc_conversions=0, adc_count=0,
                                        cells_physical=1000), physics, readout_bits=6)
    assert _term(pricing, "energy_dynamic_mj") is not None and _term(pricing, "chip_area_mm2") is not None


def _law(lo=2, hi=8, anchor=8):
    return {"form": "linear_plus_exponential", "anchor_bits": anchor, "bits_range": [lo, hi],
            "k1": {"nominal": 100.0, "unit": "fJ"}, "k2": {"nominal": 0.001, "unit": "fJ"},  # 100 fJ and 1 aJ
            "citation": "houshmand2023benchmarking Eq. 8", "derivation": "hand law"}


def test_the_energy_law_prices_a_resolution_the_table_does_not_state():
    physics = _profile({"rows": [_row(8, e=E8, a=A8)], "energy_law": _law()})
    at8 = _adc_energy(price_absolute(_q(), physics, readout_bits=8))
    at6 = _adc_energy(price_absolute(_q(), physics, readout_bits=6))
    shape = lambda b: 100e-15 * b + 1e-18 * 4 ** b  # noqa: E731
    e6 = E8 * shape(6) / shape(8)
    assert at8.nominal - at6.nominal == pytest.approx(100 * (E8 - e6) * 1e-12 * 1e3, rel=1e-9)
    area = _refusal(price_absolute(_q(), physics, readout_bits=6), "chip_area_mm2")
    assert area is not None and "area_per_adc" in area  # the law prices energy only


def test_outside_the_laws_range_refuses_by_name():
    physics = _profile({"rows": [_row(8, e=E8, a=A8)], "energy_law": _law(lo=4, hi=8)})
    reason = _refusal(price_absolute(_q(), physics, readout_bits=3), "energy_per_inference_mj")
    assert reason is not None and "4 to 8" in reason


def test_the_converter_round_trips_and_stays_out_of_a_profile_without_it():
    from mimarsinan.deployment_record.platform_physics import PlatformPhysics
    plain = get_platform_physics("generic_estimated_22nm")
    assert "converter" not in plain.to_dict()
    physics = _profile({"rows": [_row(8, e=E8, t=0.781, a=A8)], "energy_law": _law()})
    again = PlatformPhysics.from_dict(physics.to_dict())
    assert again.converter == physics.converter and again.to_dict() == physics.to_dict()


@pytest.mark.parametrize("bad, match", [
    ({"rows": [{"bits": 8, "e_mac": {"nominal": 1.0, "unit": "pJ"}}]}, "e_mac"),
    ({"rows": [_row(8, e=E8), _row(8, e=E4)]}, "twice"),
    ({"rows": [_row(8, a=A8)], "energy_law": _law(anchor=8)}, "anchor"),
    ({"rows": [_row(0, e=E8)]}, "bits"),
])
def test_an_inadmissible_converter_declaration_refuses_by_name(bad, match):
    from mimarsinan.deployment_record.platform_physics.converter import ConverterDeclarationError
    with pytest.raises(ConverterDeclarationError, match=match):
        _profile(bad)


def test_the_shipped_isaac_like_profile_states_its_converter_at_its_own_resolution():
    physics = get_platform_physics("isaac_like")
    converter = physics.converter
    assert converter is not None and converter.bits == (8,)
    row = converter.rows[8]
    for key in ("e_adc_conversion", "t_adc_conversion", "area_per_adc"):
        assert row[key].nominal == physics.constants[key].nominal and row[key].citation == physics.constants[key].citation
    pricing = price_absolute(_q(), physics, readout_bits=4)
    assert "area_per_adc at readout_law.adc_bits 4" in _refusal(pricing, "chip_area_mm2")
    assert price_absolute(_q(), physics, readout_bits=8) == price_absolute(_q(), physics)


#: [resume] The ADC energy model houshmand2023benchmarking Eq. 8 (p.5): E_ADC = (k1 ADCres + k2 4^ADCres) V^2 per
#: conversion, k1 = 100 fJ, k2 = 1 aJ, after its reference [5] (Murmann, TVLSI 29(1), 2021).
def _murmann(bits):
    return 100e-15 * bits + 1e-18 * 4.0 ** bits


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_the_shipped_isaac_like_profile_scales_its_converter_energy_by_the_cited_law(bits):
    """W34's resolutions below ISAAC's own 8-bit ADC: the energy per conversion is the 8-bit row scaled by Eq. 8's
    shape (the supply term cancels at one operating point), each corner of the row's band; the area per converter is
    stated at 8 bits only, so the area headline refuses by name; 8 bits is the flat row, unchanged."""
    physics = get_platform_physics("isaac_like")
    law = physics.converter.energy_law
    assert law is not None and law.anchor_bits == 8 and law.bits_range == (4, 8)
    assert "houshmand2023benchmarking" in law.citation and "murmann2021mixed" in law.citation
    from mimarsinan.deployment_record.platform_physics.converter import resolve
    priced, refused = resolve(physics, bits)
    flat = physics.constants["e_adc_conversion"]
    got = priced.constants["e_adc_conversion"]
    ratio = _murmann(bits) / _murmann(8)
    assert got.nominal == pytest.approx(flat.nominal * ratio, rel=1e-12)
    assert got.high == pytest.approx(flat.high * ratio, rel=1e-12) and got.evidence_kind == "derived"
    assert set(refused) == {"t_adc_conversion", "area_per_adc"}
    pricing = price_absolute(_q(), physics, readout_bits=bits)
    at8 = _adc_energy(price_absolute(_q(), physics, readout_bits=8))
    assert at8.nominal - _adc_energy(pricing).nominal == pytest.approx(
        100 * flat.nominal * (1 - ratio) * 1e-12 * 1e3, rel=1e-9)
    assert _refusal(pricing, "energy_per_inference_mj") is None or "adc_bits" not in _refusal(
        pricing, "energy_per_inference_mj")
    assert f"area_per_adc at readout_law.adc_bits {bits}" in _refusal(pricing, "chip_area_mm2")


def test_the_shipped_isaac_like_law_is_never_read_outside_its_range():
    physics = get_platform_physics("isaac_like")
    for bits in (3, 9, 32):
        reason = _refusal(price_absolute(_q(), physics, readout_bits=bits), "energy_per_inference_mj")
        assert reason is not None and f"readout_law.adc_bits {bits}" in reason and "4 to 8" in reason, bits


def test_the_planes_read_the_runs_adc_bits():
    from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
    from mimarsinan.deployment_record.cost.absolute.report import record_readout_bits
    platform = {"cores": [{"max_axons": 128, "max_neurons": 128, "count": 4}], "core_semantics": "mvm",
                "readout_law": {"adc_bits": 6, "range": "calibrated", "mult_bits": 24, "output_law": "identity"}}
    assert candidate_context_from_platform(platform).readout_bits == 6
    assert candidate_context_from_platform({**platform, "readout_law": None}).readout_bits is None
    from types import SimpleNamespace
    assert record_readout_bits(SimpleNamespace(identity=SimpleNamespace(platform=platform))) == 6
