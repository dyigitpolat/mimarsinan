"""[U-E4 A-1] The co-design study's chips as chip descriptions: COMPASS, VR-like and ISAAC-like.

The study (its estimator: cd_preview.py, x9_0_v3.py, cd41_chip.py; its constants: constants.md; its tile model: the
nangate45 physical-design probe, audit_v4/pd.md and pd_table.json) prices three digital SRAM-CIM tiles with exact
accumulation on its SRAM-DDR row: COMPASS 256 x 256 one-bit cells and VR-like 256 x 64 four-bit cells (both the
probe's G3 tile, 65,536 bits) and ISAAC-like 128 x 128 two-bit cells (the probe's G6 tile, 32,768 bits), each as
the toolchain's core of R x 64 four-bit weights. Each is a physics profile plus a target bundle here, its values the
study's own constants with their sources and classes. A quantity the study does not price carries a sourced value
with its class and band, never zero, and is named "not priced by the study" in its note and in the profile's notes.
Every number below is the study's (the file it comes from is named beside it), never typed from the profiles.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

import mimarsinan
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.registry import profile_description_path
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.mapping.platform.target_bundles import target_bundles
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

STUDY = ("study_compass", "study_vr_like", "study_isaac_like")
NOT_PRICED = "[not priced by the study]"
#: The study's numbers, each from the file named.
A_TILE_J = 2.6006349206349206e-09     # cd_preview.A_TILE_J["SRAM-DDR"]: one activation of a 65,536-bit tile
TILE_MM2 = {"study_compass": 0.1155, "study_vr_like": 0.1155, "study_isaac_like": 0.07677}  # cd_preview.TILE_MM2
TILE_BITS = {"study_compass": 65536, "study_vr_like": 65536, "study_isaac_like": 32768}      # cd_preview.TILE_BITS
ROWS = {"study_compass": 256, "study_vr_like": 256, "study_isaac_like": 128}                 # cd_preview.GEOMS
#: pd_probe/pd_table.json: the routed tiles' instance area, macro area per bit, standard cells, leakage.
PROBE = {"g3": (0.115456, 1.0373001098632812, 47475.1, 2.96386), "g6": (0.076766, 1.0373016357421876, 42775.7, 1.84622)}
TILE_OF = {"study_compass": "g3", "study_vr_like": "g3", "study_isaac_like": "g6"}
T_ACT, F_91, F_15 = 8, 91e6, 1.5e9    # cd41_chip.T_ACT and CTX (jiang2023a p.4; fujiwara2024344)
PORT_BITS, DDR4_BPS = 256, 36e9       # cd41_chip.PORT_DEFAULT; CTX's two DDR4 channels (wang2020shuhai Table V p.7)
DDR_IO_PJ = 1300.0 / 8                # jouppi2021ten Table 2 p.3 via constants.md sec. 5 (pi_inputs_out.json W_tile)
WRITE_FJ_PER_BYTE = (76.0, 342.0)     # constants.md sec. 5: array write per byte, 1x 20 MHz read, 2x 91 MHz read
F_FACTOR = 2.0                        # codesign.md sec. 3.3: the write-and-carry factor's admissibility end
ISAAC_EQUAL_AREA_MM2 = 0.5 * 0.1155   # cd42_common.ISAAC_BAND[1]
#: Quantities the study does not price: each declared from a sourced band (never zero), named in its note.
#: [U-E4 targeted pass, M1] e_readout_per_byte is not among them: the study counts a carried value in both directions
#: (x9_0_v3._front_mincarry), so the read-back crossing is the study's carry term, as the re-injection is.
#: [U-G G-1] The on-chip carry buffer's flat constants (its static power and area per byte) join them: the study's
#: estimator holds no on-chip buffer (its access energy over capacity is in the carry_buffer table,
#: test_study_chip_carry_buffer_ug).
UNPRICED = {"endurance_cycles", "e_core_program", "bytes_per_connectivity_entry",
            "p_static_per_core", "p_static_global", "e_noc_value_per_byte_hop", "e_inter_tile_hop",
            "e_intra_tile_packet", "t_hop", "e_sync_barrier", "t_sync_barrier", "area_per_router",
            "area_per_tile_fixed", "area_global_fixed",
            "p_carry_buffer_static_per_byte", "area_carry_buffer_per_byte"}
#: Terms the study prices inside another constant: declared at zero, the constant that holds them named.
INSIDE = {"area_per_cell_per_weight_bit": "area_per_cell", "area_per_row_driver": "area_per_neuron_logic",
          "e_core_init": "e_mac"}
#: The generic 22 nm bundle's band each of those reads (generic_estimated_22nm.json), with the named departures.
GENERIC_BAND = {"e_core_program", "bytes_per_connectivity_entry", "e_noc_value_per_byte_hop", "e_inter_tile_hop",
                "e_intra_tile_packet", "t_hop", "e_sync_barrier", "t_sync_barrier", "area_per_router",
                "area_per_tile_fixed", "area_global_fixed", "endurance_cycles"}


def _c(name: str, key: str):
    return get_platform_physics(name).constants[key]


def _band(name: str, key: str):
    value = _c(name, key)
    return value.low, value.nominal, value.high


def _rel(a, b, rel=1e-9):
    return all(math.isclose(x, y, rel_tol=rel) for x, y in zip(a, b))


@pytest.mark.parametrize("name", STUDY)
class TestEachProfile:
    def test_it_resolves_and_ships_its_description(self, name):
        physics = resolve_platform_physics(name, {})
        assert physics is not None and physics.name == name
        prose = profile_description_path(name).read_text()
        assert "not priced by the study" in prose and "## Rows" in prose

    def test_the_mac_energy_is_the_study_tile_activation_over_its_macs(self, name):
        macs = ROWS[name] * 64
        assert math.isclose(_c(name, "e_mac").nominal * 1e-12 * macs, A_TILE_J * TILE_BITS[name] / 65536,
                            rel_tol=1e-12)
        assert _c(name, "e_mac").low == _c(name, "e_mac").nominal == _c(name, "e_mac").high
        assert _c(name, "e_mac").evidence_kind == "derived" and "jiang2023a p.4" in _c(name, "e_mac").citation

    def test_a_latency_step_is_one_tile_activation_at_the_study_clocks(self, name):
        assert _rel(_band(name, "t_cycle"), (T_ACT / F_15 * 1e9, T_ACT / F_91 * 1e9, T_ACT / F_91 * 1e9))
        assert "jiang2023a p.4" in _c(name, "t_cycle").citation

    def test_the_write_port_is_the_slower_of_the_256_bit_row_and_the_ddr4_store(self, name):
        port = lambda f: 8.0 / PORT_BITS / f * 1e9  # noqa: E731
        low = max(port(F_15), 1e9 / DDR4_BPS)
        assert _rel(_band(name, "t_program_per_byte"), (low, port(F_91), port(F_91)))
        assert low == 1e9 / DDR4_BPS, "at 1.5 GHz the store binds, as the study's p_ceil states"

    def test_the_backing_store_and_the_array_write_with_the_write_and_carry_factor(self, name):
        assert _rel(_band(name, "e_dma_per_byte"), (DDR_IO_PJ, DDR_IO_PJ, F_FACTOR * DDR_IO_PJ))
        assert "jouppi2021ten Table 2 p.3" in _c(name, "e_dma_per_byte").citation
        slot = 0.5  # one 4-bit weight slot is half a byte
        assert _rel(_band(name, "e_write_per_cell"),
                    (WRITE_FJ_PER_BYTE[0] * slot, WRITE_FJ_PER_BYTE[0] * slot,
                     F_FACTOR * WRITE_FJ_PER_BYTE[1] * slot))
        assert _rel(_band(name, "e_program_per_byte"),
                    (WRITE_FJ_PER_BYTE[0] / 1e3, WRITE_FJ_PER_BYTE[0] / 1e3, F_FACTOR * WRITE_FJ_PER_BYTE[1] / 1e3))
        assert _c(name, "write_verify_iterations").nominal == 1.0

    def test_the_carry_crosses_the_store_both_ways_as_the_study_counts_it(self, name):
        """[U-E4 targeted pass, M1] The study counts a carried value in both directions (x9_0_v3._front_mincarry: 2 x
        the live elements at each cut, "both directions"; _io_cratio: one I/O crossing per direction, at the study's
        8-bit activations), so the read-back (e_readout_per_byte x carry_out_bytes) is the study's carry term as the
        re-injection (e_dma_per_byte x carry_in_bytes) is: the same DDR3/4 crossing, both under the factor f."""
        readout, inject = _c(name, "e_readout_per_byte"), _c(name, "e_dma_per_byte")
        assert NOT_PRICED not in readout.note, "the study prices the read-back direction"
        assert _band(name, "e_readout_per_byte") == _band(name, "e_dma_per_byte") == (DDR_IO_PJ, DDR_IO_PJ,
                                                                                       F_FACTOR * DDR_IO_PJ)
        for value in (readout, inject):
            assert "_front_mincarry" in value.derivation and "both directions" in value.derivation
            assert "8-bit" in value.derivation and "activation_bits" in value.derivation
        assert "jouppi2021ten Table 2 p.3" in readout.citation

    def test_the_tile_area_is_the_probe_tile_its_parts_and_the_study_rounding(self, name):
        instance, per_bit, stdcells, _leak = PROBE[TILE_OF[name]]
        cell, logic = _c(name, "area_per_cell"), _c(name, "area_per_neuron_logic")
        assert math.isclose(cell.nominal, 4 * per_bit, rel_tol=1e-12)
        assert math.isclose(logic.nominal, stdcells / 64, rel_tol=1e-9)
        tile_um2 = ROWS[name] * 64 * cell.nominal + 64 * logic.nominal
        assert math.isclose(tile_um2 / 1e6, instance, rel_tol=1e-5), "the probe's instance area"
        assert abs(tile_um2 / 1e6 / TILE_MM2[name] - 1) < 5e-4, "the study rounds it to four digits"
        assert "pd_table.json" in cell.citation and "pd_table.json" in logic.citation

    def test_every_quantity_the_study_does_not_price_is_declared_never_zero_and_named(self, name):
        physics = get_platform_physics(name)
        named = {key for key, value in physics.constants.items() if NOT_PRICED in value.note}
        assert named == UNPRICED
        for key in sorted(UNPRICED):
            value = physics.constants[key]
            assert value.low > 0, f"{key} is declared at zero"
            assert value.evidence_kind in ("published", "derived", "estimated") and value.basis
            assert key in physics.validity.notes, f"{key} is not named in the profile's notes"
        prose = profile_description_path(name).read_text()
        for key in sorted(UNPRICED):
            assert re.search(rf"\b{re.escape(key)}\b", prose), key

    def test_the_bands_borrowed_from_the_generic_bundle_are_its_own(self, name):
        generic = get_platform_physics("generic_estimated_22nm")
        for key in sorted(GENERIC_BAND):
            assert _band(name, key) == _band("generic_estimated_22nm", key), key
            assert _c(name, key).unit == generic.constants[key].unit
        assert _band(name, "p_static_global") == (0.5, 5.0, 50.0), "the generic band, its low end raised from 0"
        assert _band("generic_estimated_22nm", "p_static_global")[1:] == (5.0, 50.0)

    def test_terms_the_study_prices_inside_another_constant_are_zero_naming_that_constant(self, name):
        """No double count: the slot's macro area is per slot (area_per_cell), the row drivers are the tile's standard
        cells (area_per_neuron_logic), and the accumulator reset is part of a tile activation (e_mac)."""
        for key, holder in INSIDE.items():
            value = _c(name, key)
            assert (value.low, value.nominal, value.high) == (0.0, 0.0, 0.0), key
            assert f"inside {holder}" in value.derivation, key

    def test_static_power_per_core_ends_at_the_study_tile_leakage(self, name):
        leak_mw = PROBE[TILE_OF[name]][3]
        low, nominal, high = _band(name, "p_static_per_core")
        assert (low, nominal) == _band("generic_estimated_22nm", "p_static_per_core")[:2]
        assert math.isclose(high, leak_mw * 1e3, rel_tol=1e-12) and _c(name, "p_static_per_core").unit == "uW"

    def test_the_validity_names_the_probe_node_and_the_core(self, name):
        validity = get_platform_physics(name).validity
        assert (validity.technology_node_nm, validity.supply_v, validity.measurement_kind) == (45.0, 1.1, "mixed")
        assert (validity.geometry.axons, validity.geometry.neurons, validity.geometry.cell_bits) == (ROWS[name], 64, 4)


class TestTheThreeChips:
    def test_compass_and_vr_like_price_the_same_g3_tile(self):
        """The study prices both on the probe's G3 tile at one A_TILE_J: per weight slot every constant is equal; they
        differ in the cell organization only (four one-bit columns against one four-bit cell per weight)."""
        compass, vr = get_platform_physics("study_compass"), get_platform_physics("study_vr_like")
        assert set(compass.constants) == set(vr.constants)
        for key in compass.constants:
            assert compass.constants[key].to_dict() == vr.constants[key].to_dict(), key
        assert compass.validity.array_size_assumed != vr.validity.array_size_assumed

    def test_isaac_like_differs_from_compass_only_in_its_tile(self):
        compass, isaac = get_platform_physics("study_compass"), get_platform_physics("study_isaac_like")
        differ = {k for k in compass.constants if compass.constants[k].to_dict() != isaac.constants[k].to_dict()}
        # [U-G G-1] the buffer's static power and area per byte follow the tile's own leakage, bits and cell area
        assert differ == {"area_per_cell", "area_per_neuron_logic", "p_static_per_core",
                          "p_carry_buffer_static_per_byte", "area_carry_buffer_per_byte"}

    def test_the_isaac_like_area_band_is_the_study_one(self):
        """D111: the fakeram tile (0.07677 mm2) and, as the low end, equal area per bit (0.05775 mm2)."""
        cell, logic = _c("study_isaac_like", "area_per_cell"), _c("study_isaac_like", "area_per_neuron_logic")
        low_um2 = 128 * 64 * cell.low + 64 * logic.low
        assert math.isclose(low_um2 / 1e6, ISAAC_EQUAL_AREA_MM2, rel_tol=1e-12)
        assert logic.high == logic.nominal


class TestIsaacLikeReconciliation:
    """The shipped ``isaac_like`` profile is ISAAC itself (shafiee2016isaac, 32 nm analog ReRAM, bit-sliced ADC
    readout); the study's ISAAC-like tile lends ISAAC's geometry only (constants.md sec. 2) and is a digital SRAM tile.
    Every difference between the two is named in study_isaac_like.md, and isaac_like stays as it ships."""

    def _differences(self):
        isaac, study = get_platform_physics("isaac_like"), get_platform_physics("study_isaac_like")
        keys = set(isaac.constants) | set(study.constants)
        return {k for k in keys if (isaac.constants.get(k) and isaac.constants[k].to_dict()) !=
                (study.constants.get(k) and study.constants[k].to_dict())}

    def test_every_difference_is_named_in_the_reconciliation_table(self):
        prose = profile_description_path("study_isaac_like").read_text()
        section = prose.split("## Reconciliation with isaac_like", 1)[1].split("\n## ", 1)[0]
        named = set(re.findall(r"^\| `([a-z_0-9]+)` \|", section, flags=re.M))
        assert named == self._differences()

    def test_isaac_like_is_unchanged(self):
        isaac = get_platform_physics("isaac_like")
        assert isaac.conversion_model["model"] == "bit_sliced_crossbar"
        assert (isaac.constants["e_mac"].nominal, isaac.constants["area_per_cell"].nominal) == (0.0018311, 0.0015259)
        assert isaac.validity.technology_node_nm == 32.0
        assert get_platform_physics("study_isaac_like").conversion_model == {}


@pytest.mark.parametrize("name", STUDY)
class TestEachBundle:
    def test_the_bundle_declares_the_study_core_with_its_lanes(self, name):
        bundle = target_bundles()[name]
        values = bundle.values
        assert values["platform_physics_profile"] == name and values["core_semantics"] == "mvm"
        assert values["cores"] == [{"max_axons": ROWS[name], "max_neurons": 64, "count": 576, "has_bias": True}]
        assert values["weight_bits"] == 4
        assert values["onchip_arithmetic"] == {"classes": ["L", "E", "P"], "lanes_per_core": 1,
                                               "lane_block": "ng45_os_lane"}
        assert values["allow_coalescing"] is True and values["allow_neuron_splitting"] is True
        assert "activation_bits" not in values, "a program declares its own value grid"
        assert bundle.claim_class == "synthetic" and bundle.faithfulness.kind == "approximate"

    def test_a_point_on_the_chip_resolves_with_its_physics(self, name):
        pcfg = build_platform_constraints_resolved(merge_pipeline_config({}, {"target": name}))
        assert pcfg["platform_physics_resolved"] is not None
        assert pcfg["cores"][0]["max_axons"] == ROWS[name]

    def test_the_bundle_file_names_its_study_sources(self, name):
        data = json.loads((Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets" /
                           f"{name}.json").read_text())
        assert "cd41_chip.A_L" in data["geometry"]["provenance"] and "arith_toggle.md" in data["notes"]
