"""[U-G G-1] A profile's on-chip carry buffer over its declared capacity: the buffer constants at carry_buffer_bytes.

An SRAM's access energy depends on its size (jouppi2021ten Table 2 prints 8 KB, 32 KB and 1 MB SRAMs at 45 nm and 7
nm, 10, 20 and 100 pJ per 64-bit access at 45 nm), so a chip that can host a carry buffer states its constants as a
table over capacity: rows of ``bytes`` with any of ``e_carry_buffer_per_byte``, ``p_carry_buffer_static_per_byte``
and ``area_carry_buffer_per_byte`` (each with its evidence, as a profile constant); a buffer constant that holds at
every capacity (the bitcells' leakage and area per byte) is stated flat in the profile's constants instead, never in
both places. The pricer reads the run's declared capacity (``carry_buffer_bytes``, the quantity
``carry_buffer_capacity_bytes`` on both planes): the row at that capacity prices the buffer; a capacity the table
does not state refuses by name every headline whose buffer work exists; a run that declares no buffer, or a profile
without the table, prices exactly as before. Before U-G a chip could state one access energy for every capacity, so a
1 MiB buffer and a 32 KiB buffer of one chip carried the same energy per byte.

The new interface is imported inside each test so that each test fails on its own on a tree without it.
"""

from __future__ import annotations

import math

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.platform_physics import apply_overrides, get_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

QUANTITY = "carry_buffer_capacity_bytes"
ACCESS, STATIC, AREA = "e_carry_buffer_per_byte", "p_carry_buffer_static_per_byte", "area_carry_buffer_per_byte"
SMALL, LARGE = 32768, 1048576
E_SMALL, E_LARGE = 2.5, 12.5  # pJ per byte access in the hand table
HELD = 64  # carried bytes the buffer holds (each written once, read once: carry_onchip_bytes counts both)
#: 1,000 steps at the generic profile's 1,000 ns cycle: a 1 ms priced latency, no host work.
BASE = {"synaptic_events": 1e6, "latency_steps": 1000, "host_macs": 0, "compute_op_count": 0, "cores_physical": 4,
        "cells_physical": 4 * 65536, "carry_out_bytes": 16, "carry_in_bytes": 24, "carry_onchip_bytes": HELD}
FLAT = {STATIC: {"nominal": 2.0, "unit": "nW", "evidence_kind": "estimated", "note": "test: a cited buffer leakage"},
        AREA: {"nominal": 0.5, "unit": "um^2", "evidence_kind": "estimated", "note": "test: a cited buffer cell"}}


def _c(nominal, unit, note="test: hand table"):
    return {"nominal": nominal, "unit": unit, "evidence_kind": "published", "citation": note}


def _row(capacity, **constants):
    return {"bytes": capacity, **constants}


TABLE = {"rows": [_row(SMALL, **{ACCESS: _c(E_SMALL, "pJ")}), _row(LARGE, **{ACCESS: _c(E_LARGE, "pJ")})],
         "note": "test: a hand table"}


def _physics(table=TABLE, flat=FLAT):
    """generic_estimated_22nm with the hand buffer's flat constants (an operator's declaration) and ``table``."""
    from mimarsinan.deployment_record.platform_physics.loader import with_carry_buffer
    return with_carry_buffer(apply_overrides(get_platform_physics("generic_estimated_22nm"), flat), table)


def _q(capacity=None, **extra) -> Quantities:
    values = {**BASE, **extra, **({} if capacity is None else {QUANTITY: capacity})}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _terms(pricing):
    return {t.name: t for t in pricing.terms}


def _refusals(pricing):
    return {r.name: r.reason for r in pricing.refusals}


def test_a_profile_without_a_carry_buffer_table_prices_as_before():
    physics = apply_overrides(get_platform_physics("generic_estimated_22nm"), {**FLAT, ACCESS: _c(4.0, "pJ")})
    assert physics.carry_buffer is None
    priced, unstated = physics.at_carry_buffer(SMALL)
    assert priced == physics and unstated == {}


def test_no_declared_buffer_reads_no_row():
    physics = _physics()
    priced, unstated = physics.at_carry_buffer(None)
    assert priced == physics and unstated == {} and not physics.has(ACCESS)
    plain = {k: v for k, v in BASE.items() if k != "carry_onchip_bytes"}
    q = Quantities({k: QuantityValue(float(v), "static") for k, v in plain.items()})
    assert price_absolute(q, physics) == price_absolute(q, apply_overrides(get_platform_physics(
        "generic_estimated_22nm"), FLAT))


def test_the_row_at_the_declared_capacity_prices_the_buffer_access():
    physics = _physics()
    small, large = (_terms(price_absolute(_q(c), physics)) for c in (SMALL, LARGE))
    for terms in (small, large):
        assert ACCESS in terms["energy_dynamic_mj"].source and "energy_per_inference_mj" in terms
    # the dynamic energies differ by the held bytes at the two rows' access energies, in mJ
    diff = large["energy_dynamic_mj"].value - small["energy_dynamic_mj"].value
    assert math.isclose(diff, HELD * (E_LARGE - E_SMALL) * 1e-12 * 1e3, rel_tol=1e-9)
    priced, unstated = physics.at_carry_buffer(SMALL)
    assert priced.constants[ACCESS].nominal == E_SMALL and unstated == {}


def test_a_capacity_the_table_does_not_state_refuses_the_buffer_energy_by_name():
    pricing = price_absolute(_q(65536), _physics())
    refusals = _refusals(pricing)
    for name in ("energy_per_inference_mj", "energy_dynamic_mj"):
        reason = refusals.get(name)
        assert reason is not None and "carry_buffer_bytes 65536" in reason and "(32768, 1048576)" in reason, name
    assert "energy_dynamic_mj" not in _terms(pricing)
    assert "chip_area_mm2" in _terms(pricing), "the flat area per byte holds at every capacity"


def test_a_buffer_that_holds_no_carried_byte_refuses_nothing_outside_the_table():
    terms = _terms(price_absolute(_q(65536, carry_onchip_bytes=0), _physics()))
    assert "energy_dynamic_mj" in terms and "energy_per_inference_mj" in terms


def test_a_row_may_state_the_static_power_and_area_at_its_own_capacity():
    rows = {"rows": [_row(SMALL, **{ACCESS: _c(E_SMALL, "pJ"), STATIC: _c(2.0, "nW"), AREA: _c(0.5, "um^2")}),
                     _row(LARGE, **{ACCESS: _c(E_LARGE, "pJ"), STATIC: _c(3.0, "nW"), AREA: _c(0.25, "um^2")})]}
    physics = _physics(rows, flat={})
    plain = _terms(price_absolute(_q(), physics))
    small, large = (_terms(price_absolute(_q(c), physics)) for c in (SMALL, LARGE))
    assert math.isclose(small["chip_area_mm2"].value - plain["chip_area_mm2"].value, 0.5e-12 * SMALL * 1e6)
    assert math.isclose(large["chip_area_mm2"].value - plain["chip_area_mm2"].value, 0.25e-12 * LARGE * 1e6)
    assert large["energy_static_mj"].value > small["energy_static_mj"].value
    refusals = _refusals(price_absolute(_q(65536, carry_onchip_bytes=0), physics))
    assert "carry_buffer_bytes 65536" in refusals["energy_per_inference_mj"] and STATIC in refusals[
        "energy_per_inference_mj"]
    assert "carry_buffer_bytes 65536" in refusals["chip_area_mm2"] and AREA in refusals["chip_area_mm2"]


def test_an_operators_flat_override_stands_at_every_capacity():
    physics = apply_overrides(_physics(), {ACCESS: _c(7.0, "pJ", "operator departure")})
    assert physics.carry_buffer is not None, "an override keeps the profile's table"
    for capacity in (SMALL, LARGE, 65536):
        priced, unstated = physics.at_carry_buffer(capacity)
        assert priced.constants[ACCESS].nominal == 7.0 and unstated == {}, capacity


def test_the_table_round_trips_and_stays_out_of_a_profile_without_it():
    from mimarsinan.deployment_record.platform_physics import PlatformPhysics
    plain = get_platform_physics("generic_estimated_22nm")
    assert "carry_buffer" not in plain.to_dict()
    physics = _physics()
    again = PlatformPhysics.from_dict(physics.to_dict())
    assert again.carry_buffer == physics.carry_buffer and again.to_dict() == physics.to_dict()
    assert physics.carry_buffer.capacities == (SMALL, LARGE)


def test_a_profile_declares_its_table_beside_its_flat_constants():
    import json

    from mimarsinan.deployment_record.platform_physics import profile_from_dict, profiles_dir
    body = json.loads((profiles_dir() / "generic_estimated_22nm.json").read_text())
    body["constants"] = {**body["constants"], **FLAT}
    body["carry_buffer"] = TABLE
    physics = profile_from_dict(body)
    assert physics.carry_buffer.capacities == (SMALL, LARGE) and physics.has(STATIC) and not physics.has(ACCESS)
    body["constants"][ACCESS] = _c(4.0, "pJ")
    from mimarsinan.deployment_record.platform_physics.profile import CarryBufferDeclarationError
    with pytest.raises(CarryBufferDeclarationError, match=ACCESS):
        profile_from_dict(body)


@pytest.mark.parametrize("bad, match", [
    ({"rows": [_row(SMALL, e_mac=_c(1.0, "pJ"))]}, "e_mac"),
    ({"rows": [_row(SMALL, **{ACCESS: _c(E_SMALL, "pJ")}), _row(SMALL, **{ACCESS: _c(E_LARGE, "pJ")})]}, "twice"),
    ({"rows": [_row(0, **{ACCESS: _c(E_SMALL, "pJ")})]}, "bytes"),
    ({"rows": [_row(True, **{ACCESS: _c(E_SMALL, "pJ")})]}, "bytes"),
    ({"rows": [_row(SMALL)]}, "no buffer constant"),
    ({"rows": []}, "no row"),
    ({"rows": TABLE["rows"], "law": {}}, "unknown fields"),
    ({"rows": [_row(SMALL, **{STATIC: _c(2.0, "nW")})]}, STATIC),
])
def test_an_inadmissible_carry_buffer_declaration_refuses_by_name(bad, match):
    from mimarsinan.deployment_record.platform_physics.profile import CarryBufferDeclarationError
    with pytest.raises(CarryBufferDeclarationError, match=match):
        _physics(bad)
