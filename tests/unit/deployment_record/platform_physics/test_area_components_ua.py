"""[U-A A-1] Chip area modeled from declared components (owner instruction of 2026-10-28).

A chip's area is its per-core blocks, which grow with rows, columns, bits and the core count, plus its chip-level
blocks, which do not: the host link PHY by link type and lane count, a memory PHY where the chip owns a DRAM
interface, the global controller, clock generation and pads. Each block is an area component with a sourced area at
its own node and a declared node-scaling rule (logic as the node ratio squared; interface PHYs, pads and analog
blocks weaker than logic: unscaled nominal, up to the logic rule). The chip-level blocks a chip declares sum to its
``area_global_fixed``; a per-core row may name a block. Every number below is the source's, the file it comes from
named beside it, never read from the component files.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics import area_components as AC
from mimarsinan.deployment_record.platform_physics.loader import profile_from_dict
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.registry import profile_path
from mimarsinan.mapping.platform.target_bundles import target_bundles
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

#: jouppi2021ten Table 1 p.2: the TPUv1 die is below 330 mm2 at 28 nm. jouppi2017indatacenter Fig. 2 p.3 prints its
#: floor plan in whole percents: PCIe interface 3% (a PCIe Gen3 x16 link, p.3), each of two DRAM ports 3%, misc I/O 1%.
TPU1_DIE, TPU1_NODE, PCIE_LANES = 330.0, 28.0, 16
#: shafiee2016isaac Table I p.8: 4 HyperTransport links 22.88 mm2; tile 0.372 mm2; 168 tiles 62.5 mm2; chip 85.4 mm2.
HT_LINKS_MM2, ISAAC_TILE, ISAAC_TILES, ISAAC_TILES_MM2, ISAAC_CHIP = 22.88, 0.372, 168, 62.5, 85.4
#: final_units/ua/probes/orfs/<design>/6_report.json, finish__design__instance__area (ORFS nangate45, image ebc8142d).
ROUTER_UM2, IBEX_UM2 = 22308.6, 29357.1
#: shao2019simba Table 3 p.5 (16 nm synthesis area): the per-PE NoC router 19K um2, the RISC-V processor 109K um2.
SIMBA_ROUTER_UM2, SIMBA_RISCV_UM2, SIMBA_NODE = 19000.0, 109000.0, 16.0
#: goel2010a abstract: a digital PLL in 45 nm SOI of 0.111 mm2.
PLL_MM2 = 0.111
STUDY = ("study_compass", "study_vr_like", "study_isaac_like")
#: U-J's edge host description (final_units/uj; its accelerator_attach) and the study's two DDR4 channels
#: (cd41_chip.CTX: '2 DDR4 channels 36 GB/s', wang2020shuhai Table V p.7, the Alveo U280's own DDR4).
HOST, ATTACH, DDR_CHANNELS = "jetson_orin_nano_8gb", "PCIe Gen3 x4", 2


def _close(a, b, rel=1e-9):
    return all(math.isclose(x, y, rel_tol=rel) for x, y in zip(a, b))


class TestEachComponentsValueAndSource:
    def test_every_shipped_component_is_sourced_and_scalable(self):
        names = AC.component_names()
        assert set(names) >= {"pcie_gen3_phy_lane_tpuv1", "ddr_x64_port_tpuv1", "misc_io_tpuv1",
                              "hypertransport_link_isaac", "ibex_core_ng45", "lc_dpll_goel2010_45nm",
                              "dynamic_node_router_ng45"}
        for name in names:
            component = AC.load_component(name)
            assert component.area["evidence_kind"] in AC.SOURCED, name
            assert component.scaling in AC.scaling_rules() and component.kind in AC.KINDS, name
            assert component.area.get("citation") or component.area.get("note"), name

    @pytest.mark.parametrize("name, share, per", [("pcie_gen3_phy_lane_tpuv1", 0.03, PCIE_LANES),
                                                  ("ddr_x64_port_tpuv1", 0.03, 1), ("misc_io_tpuv1", 0.01, 1)])
    def test_the_tpuv1_blocks_are_their_floorplan_share_of_the_die_bound(self, name, share, per):
        component = AC.load_component(name)
        expect = [(share + d) * TPU1_DIE / per * 1e6 for d in (-0.005, 0.0, 0.005)]  # whole percents printed
        assert _close(component.band_um2(), expect)
        assert component.technology_node_nm == TPU1_NODE and component.scaling == "io_phy"
        assert "jouppi2017indatacenter" in component.area["citation"] and "jouppi2021ten" in component.area["citation"]

    def test_the_pcie_lane_serves_pcie_gen3_per_lane(self):
        component = AC.load_component("pcie_gen3_phy_lane_tpuv1")
        assert (component.kind, component.per, component.link_family) == ("link_phy", "lane", "PCIe Gen3")

    def test_the_hypertransport_link_is_a_quarter_of_isaacs_four(self):
        component = AC.load_component("hypertransport_link_isaac")
        assert _close(component.band_um2(), [HT_LINKS_MM2 / 4 * 1e6] * 3)
        assert (component.technology_node_nm, component.per) == (32.0, "link")
        assert "shafiee2016isaac Table I" in component.area["citation"]

    @pytest.mark.parametrize("name, measured, simba", [("dynamic_node_router_ng45", ROUTER_UM2, SIMBA_ROUTER_UM2),
                                                       ("ibex_core_ng45", IBEX_UM2, SIMBA_RISCV_UM2)])
    def test_the_measured_blocks_hold_at_45_nm_with_simbas_block_as_the_high_end(self, name, measured, simba):
        component = AC.load_component(name)
        assert component.area["evidence_kind"] == "measured" and component.technology_node_nm == 45.0
        assert _close(component.band_um2(), [measured, measured, simba * (45.0 / SIMBA_NODE) ** 2])
        assert "ebc8142d" in component.area["note"] and "shao2019simba Table 3" in component.area["note"]

    def test_the_pll_is_the_published_45_nm_one(self):
        component = AC.load_component("lc_dpll_goel2010_45nm")
        assert _close(component.band_um2(), [PLL_MM2 * 1e6] * 3) and component.kind == "clock"
        assert component.area["evidence_kind"] == "published" and "goel2010a" in component.area["citation"]


class TestTheNodeScalingRules:
    def test_logic_scales_as_the_node_ratio_squared(self):
        assert _close(AC.scaling_rules()["logic"].factors(28.0, 45.0), [(45 / 28) ** 2] * 3)

    def test_interface_phys_scale_weaker_than_logic_in_either_direction(self):
        rule = AC.scaling_rules()["io_phy"]
        assert _close(rule.factors(28.0, 45.0), [1.0, 1.0, (45 / 28) ** 2])
        assert _close(rule.factors(28.0, 22.0), [(22 / 28) ** 2, 1.0, 1.0])
        assert "jouppi2021ten" in rule.basis and "jouppi2017indatacenter" in rule.basis

    def test_a_component_at_the_chip_node_is_unscaled_and_named(self):
        band, note = AC.scaled_um2(AC.load_component("ibex_core_ng45"), 45.0)
        assert _close(band, AC.load_component("ibex_core_ng45").band_um2()) and "unscaled" in note

    def test_a_component_beside_a_chip_with_no_node_is_refused_by_name(self):
        with pytest.raises(AC.AreaComponentError, match="pcie_gen3_phy_lane_tpuv1.*no technology node"):
            AC.scaled_um2(AC.load_component("pcie_gen3_phy_lane_tpuv1"), None)


class TestRefusalsByName:
    def test_an_unsourced_component_is_refused_by_name(self):
        component = AC.load_component("misc_io_tpuv1")
        bare = replace(component, name="bare_estimate",
                       area={"nominal": 1.0, "unit": "mm^2", "evidence_kind": "estimated", "note": "a guess"})
        with pytest.raises(AC.AreaComponentError, match="'bare_estimate' is unsourced"):
            AC.check_sourced(bare)

    def test_an_unknown_component_is_refused_with_the_shipped_list(self):
        with pytest.raises(AC.AreaComponentError, match="'no_such_block' is not shipped"):
            AC.load_component("no_such_block")

    def test_a_scalar_beside_chip_components_is_refused(self):
        with pytest.raises(AC.AreaComponentError, match="counted? it twice|count it twice"):
            AC.resolve_area_declarations({"area_global_fixed": {"nominal": 1.0, "evidence_kind": "derived",
                                                                 "derivation": "x"}},
                                         [{"component": "misc_io_tpuv1", "count": 1}], 45.0)

    def test_a_link_phy_counted_off_its_lanes_or_family_is_refused(self):
        lane = {"component": "pcie_gen3_phy_lane_tpuv1", "link": {"host": HOST, "attach": ATTACH}}
        with pytest.raises(AC.AreaComponentError, match="needs count 4, the chip declares 2"):
            AC.chip_components_of([{**lane, "count": 2}])
        with pytest.raises(AC.AreaComponentError, match="names no '<family> x<lanes>'"):
            AC.chip_components_of([{**lane, "count": 4, "link": {"host": HOST, "attach": "PCIe Gen4 x4"}}])
        assert AC.chip_components_of([{**lane, "count": 4}])[0].count == 4

    def test_a_per_core_block_is_not_a_chip_component_and_back(self):
        with pytest.raises(AC.AreaComponentError, match="not a chip-level one"):
            AC.chip_components_of([{"component": "dynamic_node_router_ng45", "count": 1}])
        with pytest.raises(AC.AreaComponentError, match="chip-level"):
            AC.component_row("area_per_router", {"component": "misc_io_tpuv1"}, 45.0)


def _by_hand_fixed(node: float, decls):
    """The chip-level sum at ``node`` from the sources above (TPUv1 at 28 nm under io_phy, the 45 nm blocks)."""
    io = sorted([1.0, (node / TPU1_NODE) ** 2])
    tpu = {"pcie_gen3_phy_lane_tpuv1": (0.03, PCIE_LANES), "ddr_x64_port_tpuv1": (0.03, 1), "misc_io_tpuv1": (0.01, 1)}
    total = [0.0, 0.0, 0.0]
    for name, count in decls:
        if name in tpu:
            share, per = tpu[name]
            ends = [(share - 0.005) * TPU1_DIE / per * io[0], share * TPU1_DIE / per, (share + 0.005) * TPU1_DIE / per * io[1]]
        elif name == "ibex_core_ng45":
            ends = [IBEX_UM2 / 1e6, IBEX_UM2 / 1e6, SIMBA_RISCV_UM2 * (45 / SIMBA_NODE) ** 2 / 1e6]
        else:
            ends = [PLL_MM2] * 3
        total = [t + count * e for t, e in zip(total, ends)]
    return total


STUDY_DECLS = [("pcie_gen3_phy_lane_tpuv1", 4), ("ddr_x64_port_tpuv1", DDR_CHANNELS), ("ibex_core_ng45", 1),
               ("lc_dpll_goel2010_45nm", 1), ("misc_io_tpuv1", 1)]


@pytest.mark.parametrize("name", STUDY)
class TestTheStudyChips:
    def test_the_chip_level_blocks_are_the_edge_link_the_own_store_controller_clock_and_pads(self, name):
        decls = get_platform_physics(name).chip_components
        assert [(d.component, d.count) for d in decls] == STUDY_DECLS
        link = decls[0].link
        assert (link["host"], link["attach"]) == (HOST, ATTACH), "U-J's link declaration, not a second one"
        assert "DDR4" in decls[1].role and "36 GB/s" in decls[1].note, "the study's own two DDR4 channels"

    def test_the_fixed_area_is_their_sum_at_45_nm_priced_by_hand(self, name):
        value = get_platform_physics(name).constants["area_global_fixed"]
        assert value.evidence_kind == "derived" and value.unit == "mm^2" and not value.overridden
        assert _close((value.low, value.nominal, value.high), _by_hand_fixed(45.0, STUDY_DECLS))
        assert all(c in value.derivation for c, _n in STUDY_DECLS)

    def test_the_router_is_the_measured_block_and_the_tile_holds_its_own_control(self, name):
        physics = get_platform_physics(name)
        router = physics.constants["area_per_router"]
        assert _close((router.low, router.nominal, router.high),
                      [ROUTER_UM2, ROUTER_UM2, SIMBA_ROUTER_UM2 * (45 / SIMBA_NODE) ** 2])
        assert router.evidence_kind == "derived" and "dynamic_node" in router.derivation
        fixed = physics.constants["area_per_tile_fixed"]
        assert (fixed.low, fixed.nominal, fixed.high) == (0.0, 0.0, 0.0)
        for holder in ("area_per_neuron_logic", "area_per_router", "chip_components"):
            assert holder in fixed.derivation, f"the tile overhead's holder {holder} is named"

    def test_no_area_row_is_an_unsourced_estimate_any_more(self, name):
        physics = get_platform_physics(name)
        assert {k for k, v in physics.constants.items()
                if k.startswith("area") and v.evidence_kind == "estimated"} == set()
        assert AC.fixed_area_refusal(physics) is None


class TestIsaacLikeReproducesTableI:
    def test_the_four_links_are_its_fixed_area(self):
        physics = get_platform_physics("isaac_like")
        assert [(d.component, d.count) for d in physics.chip_components] == [("hypertransport_link_isaac", 4)]
        value = physics.constants["area_global_fixed"]
        assert math.isclose(value.nominal, HT_LINKS_MM2, rel_tol=1e-12) and value.evidence_kind == "derived"

    def test_one_tile_from_its_components_is_table_is_tile(self):
        """96 arrays per tile (12 IMAs x 8 arrays of 128 x 128): cells, rows (DAC and input register), ADCs,
        columns (sample-and-hold, shift-and-add, output register), the router's quarter and the tile block."""
        c = {k: v.nominal for k, v in get_platform_physics("isaac_like").constants.items() if k.startswith("area")}
        arrays, rows, cols = 96, 128, 128
        tile_um2 = (arrays * rows * cols * c["area_per_cell"] + arrays * rows * c["area_per_row_driver"]
                    + arrays * c["area_per_adc"] + arrays * cols * c["area_per_neuron_logic"]
                    + c["area_per_router"] + c["area_per_tile_fixed"])
        assert abs(tile_um2 / 1e6 - ISAAC_TILE) < 0.0005, "Table I's 0.372 mm2 tile (three digits)"
        assert abs(ISAAC_TILES * tile_um2 / 1e6 - ISAAC_TILES_MM2) < 0.1, "Table I's 62.5 mm2 of tiles"
        assert abs(ISAAC_TILES * tile_um2 / 1e6 + HT_LINKS_MM2 - ISAAC_CHIP) < 0.1, "Table I's 85.4 mm2 chip"

    def test_the_bundle_declares_isaacs_tile_of_96_arrays(self):
        """Table I's chip: 168 tiles (14 x 12) of 96 arrays, so the resolved floorplan's NoC tile is ISAAC's tile."""
        assert target_bundles()["isaac_like"].values["cores_per_tile"] == 96
        pcfg = build_platform_constraints_resolved(merge_pipeline_config({}, {"target": "isaac_like"}))
        rows, cols = pcfg["tile_grid_rows_resolved"], pcfg["tile_grid_cols_resolved"]
        assert pcfg["cores_per_tile_resolved"] == 96 and rows * cols == ISAAC_TILES and {rows, cols} == {12, 14}


class TestTheProfileShape:
    def test_chip_components_round_trip_and_stay_absent_where_undeclared(self):
        physics = get_platform_physics("study_compass")
        again = PlatformPhysics.from_dict(physics.to_dict())
        assert again.chip_components == physics.chip_components and again == physics
        assert "chip_components" not in get_platform_physics("generic_estimated_22nm").to_dict()

    def test_a_profile_resolves_a_component_row_at_its_own_node(self):
        import json
        data = json.loads(profile_path("study_compass").read_text())
        assert data["constants"]["area_per_router"] == {"component": "dynamic_node_router_ng45",
                                                         "note": data["constants"]["area_per_router"]["note"]}
        moved = {**data, "validity": {**data["validity"], "technology_node_nm": 90.0}}
        router = profile_from_dict(moved).constants["area_per_router"]
        assert math.isclose(router.nominal, ROUTER_UM2 * (90 / 45) ** 2, rel_tol=1e-12)
