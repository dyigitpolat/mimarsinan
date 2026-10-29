"""[U-G G-1] The study chips declare an on-chip carry buffer (U-F4's ``carry_buffer_bytes`` component) generically.

The levers scout on the study chips (RA/11_wave2/levers_study/scout.md sections 2, 6, 8 and 9) found the carry buffer
the strongest lever. Each of the three study chip descriptions now states the buffer's physics, each constant with its
class and band, and none of it is priced by the study (its estimator holds no on-chip buffer):
- the access energy per byte over capacity (``carry_buffer`` table rows), from jouppi2021ten Table 2 p.3 at 45 nm, the
  study chips' node: 8 KB, 32 KB and 1 MB SRAMs at 10, 20 and 100 pJ per 64-bit access (the 1 MB row a single bank,
  its footnote 1), over 8 bytes per access; a point value at each capacity (one 45 nm figure per size);
- the area per byte from the chip's own cells per bit: the cell's area at its full width (``area_per_cell`` + cell_bits
  x ``area_per_cell_per_weight_bit``) over its cell_bits, x 8 (a study cell is a 4-bit slot, not one bit);
- the static power per byte from the chip's declared leakage: ``p_static_per_core`` over the bit cells of its declared
  core (axons x neurons x cell_bits), x 8, the whole core leakage charged to its bitcells (an upper bound, so stated).
Bitcells only: the buffer's periphery is not priced. Every number below is the source's (named beside it), never read
from the profiles under test.
"""

from __future__ import annotations

import math
import re

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.platform_physics import PlatformPhysics, available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.registry import profile_description_path
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

STUDY = ("study_compass", "study_vr_like", "study_isaac_like")
NOT_PRICED = "[not priced by the study]"
ACCESS, STATIC, AREA = "e_carry_buffer_per_byte", "p_carry_buffer_static_per_byte", "area_carry_buffer_per_byte"
#: jouppi2021ten Table 2 p.3, 45 nm column, pJ per 64-bit access ("Memory is pJ per 64-bit access").
JOUPPI_45NM_PJ_PER_ACCESS = {8192: 10.0, 32768: 20.0, 1048576: 100.0}
#: pd_probe/pd_table.json: the routed tiles' macro area per bit (um^2) and leakage (mW), as test_study_chips_ue4.
MACRO_UM2_PER_BIT = {"g3": 1.0373001098632812, "g6": 1.0373016357421876}
LEAK_MW = {"g3": 2.96386, "g6": 1.84622}
TILE_OF = {"study_compass": "g3", "study_vr_like": "g3", "study_isaac_like": "g6"}
TILE_BITS = {"study_compass": 65536, "study_vr_like": 65536, "study_isaac_like": 32768}  # cd_preview.TILE_BITS
#: The generic 22 nm band's low and nominal leakage per core (uW), the study profiles' p_static_per_core ends.
GENERIC_LEAK_UW = (1.0, 50.0)
HELD = 1000  # carried bytes the buffer holds in the pricing checks


def _q(capacity=None, held=HELD) -> Quantities:
    values = {"synaptic_events": 1e6, "latency_steps": 1000, "host_macs": 0, "compute_op_count": 0,
              "cores_physical": 144, "cells_physical": 144 * 16384, "carry_out_bytes": 16, "carry_in_bytes": 24,
              "carry_onchip_bytes": held, **({} if capacity is None else {"carry_buffer_capacity_bytes": capacity})}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _terms(pricing):
    return {t.name: t for t in pricing.terms}


@pytest.mark.parametrize("name", STUDY)
class TestEachStudyChip:
    def test_the_access_energy_per_byte_is_jouppi_table_2_at_45_nm_per_capacity(self, name):
        physics = get_platform_physics(name)
        table = physics.carry_buffer
        assert table is not None and table.capacities == tuple(sorted(JOUPPI_45NM_PJ_PER_ACCESS))
        assert not physics.has(ACCESS), "the access energy depends on the capacity: rows only"
        for capacity, per_access in JOUPPI_45NM_PJ_PER_ACCESS.items():
            value = table.rows[capacity][ACCESS]
            assert (value.low, value.nominal, value.high) == (per_access / 8,) * 3, capacity
            assert value.unit == "pJ" and value.evidence_kind == "derived"
            assert "jouppi2021ten Table 2" in value.citation and "p.3" in value.citation
            assert "64-bit access" in value.derivation and "45 nm" in value.derivation
            assert NOT_PRICED in value.note
        assert "single bank" in table.rows[1048576][ACCESS].derivation, "Table 2 footnote 1"

    def test_the_area_per_byte_is_the_chip_s_own_cells_per_bit(self, name):
        physics = get_platform_physics(name)
        value = physics.constants[AREA]
        per_bit = MACRO_UM2_PER_BIT[TILE_OF[name]]  # the slot's 4 bits of macro, per bit
        assert (value.low, value.nominal, value.high) == pytest.approx((8 * per_bit,) * 3, rel=1e-12)
        cell, wide = physics.constants["area_per_cell"], physics.constants["area_per_cell_per_weight_bit"]
        bits = physics.validity.geometry.cell_bits
        assert math.isclose(value.nominal, 8 * (cell.nominal + bits * wide.nominal) / bits, rel_tol=1e-12)
        assert value.unit == "um^2" and value.evidence_kind == "derived" and NOT_PRICED in value.note
        assert "periphery" in value.note and "area_per_cell" in value.derivation

    def test_the_static_power_per_byte_is_the_whole_core_leakage_on_its_bitcells(self, name):
        physics = get_platform_physics(name)
        value = physics.constants[STATIC]
        bits = TILE_BITS[name]
        geometry = physics.validity.geometry
        assert geometry.axons * geometry.neurons * geometry.cell_bits == bits
        ends = (*GENERIC_LEAK_UW, LEAK_MW[TILE_OF[name]] * 1e3)  # uW per core
        assert (value.low, value.nominal, value.high) == pytest.approx(tuple(8 * u * 1e3 / bits for u in ends),
                                                                       rel=1e-12)
        assert value.unit == "nW" and value.evidence_kind == "estimated" and NOT_PRICED in value.note
        assert "upper bound" in value.note and "p_static_per_core" in value.note

    def test_the_buffer_is_named_where_the_profile_names_what_the_study_does_not_price(self, name):
        physics = get_platform_physics(name)
        prose = profile_description_path(name).read_text()
        for key in (ACCESS, STATIC, AREA, "carry_buffer"):
            assert key in physics.validity.notes, key
            assert re.search(rf"\b{re.escape(key)}\b", prose), key
        assert "jouppi2021ten" in prose and "upper bound" in prose

    def test_the_pricer_reads_the_row_at_the_declared_capacity(self, name):
        physics = get_platform_physics(name)
        plain = _terms(price_absolute(_q(), physics))
        small, large = (_terms(price_absolute(_q(c), physics)) for c in (32768, 1048576))
        diff = large["energy_dynamic_mj"].value - small["energy_dynamic_mj"].value
        assert math.isclose(diff, HELD * (100.0 - 20.0) / 8 * 1e-12 * 1e3, rel_tol=1e-9)
        per_byte_mm2 = 8 * MACRO_UM2_PER_BIT[TILE_OF[name]] * 1e-6
        assert math.isclose(small["chip_area_mm2"].value - plain["chip_area_mm2"].value, 32768 * per_byte_mm2,
                            rel_tol=1e-9)
        assert math.isclose(large["chip_area_mm2"].value - plain["chip_area_mm2"].value, 1048576 * per_byte_mm2,
                            rel_tol=1e-9)
        refused = {r.name: r.reason for r in price_absolute(_q(65536), physics).refusals}
        assert "carry_buffer_bytes 65536" in refused["energy_per_inference_mj"]
        assert "(8192, 32768, 1048576)" in refused["energy_per_inference_mj"]

    def test_the_resolved_physics_carries_the_table(self, name):
        physics = resolve_platform_physics(name, None)
        assert physics is not None and physics.carry_buffer == get_platform_physics(name).carry_buffer
        assert PlatformPhysics.from_dict(physics.to_dict()).carry_buffer == physics.carry_buffer


def test_compass_and_vr_like_state_the_same_buffer():
    compass, vr = get_platform_physics("study_compass"), get_platform_physics("study_vr_like")
    assert compass.carry_buffer == vr.carry_buffer
    for key in (STATIC, AREA):
        assert compass.constants[key].to_dict() == vr.constants[key].to_dict(), key


def test_no_other_chip_description_states_a_buffer():
    for name in sorted(set(available_profiles()) - set(STUDY)):
        physics = get_platform_physics(name)
        assert physics.carry_buffer is None and not any(physics.has(k) for k in (ACCESS, STATIC, AREA)), name
