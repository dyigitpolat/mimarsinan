"""[U-A A-1] The chip area the pricer states from declared components, priced by hand.

The pricer's chip area is the per-core blocks (cells, row periphery, ADCs, column logic, the router and the block of
each NoC tile, the lanes) plus the chip-level area ``area_global_fixed``: the sum of the chip's declared chip
components where it declares them, else a sourced scalar. Where neither exists the chip area is refused by name: an
absent fixed area, or an unsourced estimate on a chip that is not a projection (a projection, whose every constant is
an estimate by declaration, prices its estimate as it always did). Every number is the source's, named beside it.
"""

from __future__ import annotations

import math

import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.platform_physics import apply_overrides, get_platform_physics
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

#: pd_probe/pd_table.json g3: the routed tile's instance area and its macro area per bit and standard cells (um2).
G3_MM2, MACRO_UM2_PER_BIT, G3_STDCELL_UM2 = 0.115456, 1.0373001098632812, 47475.1
#: lane_blocks/ng45_os_lane.json area_per_lane (45 nm, the study's node): 8,202.9 / 13,047.9 / 15,000.1 um2.
LANE_UM2 = (8202.9, 13047.9, 15000.1)
#: final_units/ua/probes/orfs/dynamic_node/6_report.json; shao2019simba Table 3 p.5 (16 nm) at the logic rule.
ROUTER_UM2 = (22308.6, 22308.6, 19000.0 * (45 / 16) ** 2)
#: jouppi2017indatacenter Fig. 2 p.3 shares of the TPUv1 die below 330 mm2 (jouppi2021ten Table 1 p.2); the 45 nm
#: blocks (ibex: final_units/ua/probes/orfs/ibex/6_report.json, Simba's RISC-V 109K um2 at 16 nm; PLL goel2010a).
IO = (1.0, 1.0, (45 / 28) ** 2)
FIXED = tuple(4 * (0.03 + d) * 330 / 16 * f + 2 * (0.03 + d) * 330 * f + (0.01 + d) * 330 * f + ibex + 0.111
              for d, f, ibex in zip((-0.005, 0.0, 0.005), IO, (0.0293571, 0.0293571, 109000.0 * (45 / 16) ** 2 / 1e6)))
#: shafiee2016isaac Table I p.8.
ISAAC_TILE, ISAAC_TILES_MM2, HT_MM2, ISAAC_CHIP = 0.372, 62.5, 22.88, 85.4


def _q(**values: float) -> Quantities:
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _area(physics, **quantities):
    pricing = price_absolute(_q(**quantities), physics)
    terms = {t.name: t for t in pricing.terms}
    refusals = {r.name: r.reason for r in pricing.refusals}
    if "chip_area_mm2" in terms:
        band = terms["chip_area_mm2"].band
        return (band.low, band.nominal, band.high), terms["chip_area_mm2"].source, None
    return None, None, refusals.get("chip_area_mm2")


def _study(cores: int, tiles: int, cores_per_tile: int):
    """Quantities of ``cores`` study tiles (256 rows x 64 four-bit slots) on ``tiles`` NoC tiles."""
    return dict(cells_physical=cores * 256 * 64, weight_bits=4, axons_physical=cores * 256,
                neurons_physical=cores * 64, cores_physical=cores, tiles=tiles, lanes_per_tile=cores_per_tile)


def _close(a, b, rel=1e-9):
    return all(math.isclose(x, y, rel_tol=rel) for x, y in zip(a, b))


STUDY = resolve_platform_physics("study_compass", {}, lane_block="ng45_os_lane")


class TestTheStudyChipPricedByHand:
    def test_one_study_tile_on_its_own_noc_tile(self):
        area, source, refused = _area(STUDY, **_study(1, 1, 1))
        assert refused is None
        tile = 256 * 64 * 4 * MACRO_UM2_PER_BIT + G3_STDCELL_UM2           # the probe's instance area
        assert math.isclose(tile / 1e6, G3_MM2, rel_tol=1e-5)
        per_core = [(tile + lane + router) / 1e6 for lane, router in zip(LANE_UM2, ROUTER_UM2)]
        assert _close([a - f for a, f in zip(area, FIXED)], per_core, rel=1e-6)
        assert "global: area_global_fixed [derived" in source and "routers" in source

    def test_the_budget_chip_of_576_tiles_on_144_noc_tiles(self):
        """576 study tiles (A_L), four per NoC tile (the resolved floorplan's cores_per_tile): one router each."""
        area, _source, refused = _area(STUDY, **_study(576, 144, 4))
        tile = 256 * 64 * 4 * MACRO_UM2_PER_BIT + G3_STDCELL_UM2
        expect = [(576 * (tile + lane) + 144 * router) / 1e6 + fixed
                  for lane, router, fixed in zip(LANE_UM2, ROUTER_UM2, FIXED)]
        assert refused is None and _close(area, expect, rel=1e-6)

    def test_the_chip_level_share_is_stated_by_the_profile(self):
        fixed = STUDY.constants["area_global_fixed"]
        assert _close((fixed.low, fixed.nominal, fixed.high), FIXED, rel=1e-9)


class TestRefusedByName:
    def test_an_unsourced_fixed_area_on_a_chip_that_is_not_a_projection(self):
        bare = apply_overrides(get_platform_physics("truenorth"), {})
        estimate = bare.constants["area_global_fixed"]
        from dataclasses import replace
        physics = replace(bare, constants={**bare.constants, "area_global_fixed": replace(
            estimate, evidence_kind="estimated", citation="", derivation="", note="a band borrowed elsewhere")})
        area, _s, refused = _area(physics, cores_physical=4)
        assert area is None and "area_global_fixed is an unsourced estimate" in refused and "'silicon'" in refused

    def test_an_absent_fixed_area(self):
        physics = get_platform_physics("odin")
        from dataclasses import replace
        stripped = replace(physics, constants={k: v for k, v in physics.constants.items() if k != "area_global_fixed"})
        area, _s, refused = _area(stripped, cores_physical=1)
        assert area is None and "neither chip_components nor area_global_fixed" in refused

    def test_a_projection_prices_its_estimate_unchanged(self):
        """generic_estimated_22nm is a projection: every constant an estimate by declaration (its validity notes)."""
        physics = get_platform_physics("generic_estimated_22nm")
        assert physics.validity.measurement_kind == "projection"
        area, source, refused = _area(physics, cells_physical=65536, weight_bits=8, neurons_physical=256,
                                      axons_physical=256, cores_physical=1, tiles=1)
        fixed = physics.constants["area_global_fixed"]
        assert refused is None and fixed.evidence_kind == "estimated" and "area_global_fixed [estimated" in source


class TestIsaacLike:
    def _isaac(self, tiles: int):
        arrays = 96 * tiles   # 12 IMAs x 8 arrays of 128 x 128 two-bit cells per tile, one ADC per array
        return _area(get_platform_physics("isaac_like"), cells_physical=arrays * 128 * 128, weight_bits=2,
                     axons_physical=arrays * 128, neurons_physical=arrays * 128, adc_count=arrays,
                     cores_physical=arrays, tiles=tiles)

    def test_one_tile_and_the_chip_of_168(self):
        one, _s, refused = self._isaac(1)
        assert refused is None and abs(one[1] - HT_MM2 - ISAAC_TILE) < 0.0005
        chip, _s, _r = self._isaac(168)
        assert abs(chip[1] - HT_MM2 - ISAAC_TILES_MM2) < 0.1 and abs(chip[1] - ISAAC_CHIP) < 0.1
