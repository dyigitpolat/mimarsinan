"""The arithmetic-lane block: area, energy and latency of on-chip lanes, read only from a lane census."""

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import (
    apply_overrides,
    available_profiles,
    get_platform_physics,
)
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks

from unit.deployment_record.g3_fixtures import constant, quantities
from unit.deployment_record.printed_names import NOT_PRODUCED, named, parts, without

_LANES = merge_override_blocks({"lane": {
    "area_per_lane": constant(741.8, unit="um^2", note="routed probe lane, Nangate 45 nm (test)"),
    "e_lane_op": constant(0.15, unit="pJ", low=0.015, high=0.15, note="lane energy band (test)"),
    "t_lane_op": constant(2.0, unit="ns", note="lane pass per element (test)"),
}})
_BASE = get_platform_physics("generic_estimated_22nm")
_WITH_LANES = apply_overrides(_BASE, _LANES)


def _census(**extra):
    return quantities(cells_physical=65536, tiles=4, neurons_physical=256, axons_physical=256,
                      synaptic_events=1000, latency_steps=10, cores_physical=1, host_macs=0, **extra)


def _terms(physics, values):
    return {t.name: t for t in price_absolute(values, physics).terms}


#: [U-ENERGY class 2] The lane rows: declared over a census the view does not produce, each is named beside the
#: estimate with the plane's reason, never priced and never dropped silently.
_LANE_ROWS = ("area_per_lane", "e_lane_op", "t_lane_op")


def _lane_free(pricing):
    """Each term's dict with the lane rows' names taken out of its texts, after checking that every such name states
    "not produced on this plane": the values and the priced texts are the report's own."""
    out = []
    for term in pricing.terms:
        data = term.to_dict()
        for row in _LANE_ROWS:
            assert all(NOT_PRODUCED in n for n in named(term.source, row)), (term.name, row)
        data["source"] = without(term.source, _LANE_ROWS)
        if data["band"] is not None:
            data["band"] = {**data["band"], "basis": without(data["band"]["basis"], _LANE_ROWS)}
        out.append(data)
    return out


class TestTheFormulas:
    def test_lane_area_joins_the_chip_area(self):
        with_lanes = _terms(_WITH_LANES, _census(lanes_per_tile=8, lane_ops=5000))["chip_area_mm2"]
        without = _terms(_WITH_LANES, _census())["chip_area_mm2"]
        assert with_lanes.value - without.value == pytest.approx(741.8e-12 * 8 * 4 * 1e6, rel=1e-9)
        assert "lanes: area_per_lane" in with_lanes.source

    def test_lane_energy_is_its_row_times_the_lane_ops(self):
        with_lanes = _terms(_WITH_LANES, _census(lanes_per_tile=8, lane_ops=5000))["energy_dynamic_mj"]
        without = _terms(_WITH_LANES, _census())["energy_dynamic_mj"]
        assert with_lanes.value - without.value == pytest.approx(0.15e-12 * 5000 * 1e3, rel=1e-9)
        assert with_lanes.band.low - without.band.low == pytest.approx(0.015e-12 * 5000 * 1e3, rel=1e-9)

    def test_the_lane_pass_runs_its_lanes_in_parallel(self):
        with_lanes = _terms(_WITH_LANES, _census(lanes_per_tile=8, lane_ops=5000))["e2e_latency_s"]
        without = _terms(_WITH_LANES, _census())["e2e_latency_s"]
        assert with_lanes.value - without.value == pytest.approx(2e-9 * 5000 / (8 * 4), rel=1e-9)


class TestAbsentByDefault:
    def test_without_a_lane_census_every_report_is_unchanged(self):
        values = _census()
        assert _lane_free(price_absolute(values, _WITH_LANES)) == \
            [t.to_dict() for t in price_absolute(values, _BASE).terms]

    def test_no_shipped_profile_declares_a_lane_row(self):
        for name in available_profiles():
            assert not {"area_per_lane", "e_lane_op", "t_lane_op"} & set(get_platform_physics(name).constants)

    def test_the_rows_live_in_the_compute_group(self):
        assert {spec_for(k).group for k in ("area_per_lane", "e_lane_op", "t_lane_op")} == {"compute"}


class TestALaneCensusTheTargetCannotPrice:
    """Targeted pass: a lane census on physics without the lane rows names them on every headline it
    reaches (area and latency as energy does), never prices the lanes as free."""

    _LANE_CENSUS = dict(lanes_per_tile=8, lane_ops=5000)

    def test_the_component_area_names_the_undeclared_lane_row(self):
        area = _terms(_BASE, _census(**self._LANE_CENSUS))["chip_area_mm2"]
        assert "area_per_lane(lanes)" in area.source.split("; unpriced: ")[1]
        assert area.value == _terms(_BASE, _census())["chip_area_mm2"].value

    def test_the_footprint_area_names_the_undeclared_lane_row(self):
        truenorth = get_platform_physics("truenorth")
        area = _terms(truenorth, _census(**self._LANE_CENSUS))["chip_area_mm2"]
        assert "area_per_lane(lanes)" in area.source.split("; unpriced: ")[1]

    def test_the_latency_names_the_undeclared_lane_row(self):
        latency = _terms(_BASE, _census(**self._LANE_CENSUS))["e2e_latency_s"]
        assert "t_lane_op(lanes)" in latency.source.split("; unpriced: ")[1]

    def test_energy_still_names_its_lane_row(self):
        energy = _terms(_BASE, _census(**self._LANE_CENSUS))["energy_dynamic_mj"]
        assert "e_lane_op(lane)" in energy.source

    @pytest.mark.parametrize("physics", [_BASE, _WITH_LANES], ids=["undeclared", "declared"])
    def test_lane_ops_without_the_lanes_name_the_lane_pass(self, physics):
        latency = _terms(physics, _census(lane_ops=5000))["e2e_latency_s"]
        assert "t_lane_op(lanes; " in latency.source and "lanes_per_tile" in latency.source

    def test_lane_ops_on_no_lanes_name_the_lane_pass(self):
        latency = _terms(_WITH_LANES, _census(lane_ops=5000, lanes_per_tile=0))["e2e_latency_s"]
        assert "t_lane_op(lanes; " in latency.source

    def test_lanes_without_the_tile_count_name_the_lane_area(self):
        values = quantities(cells_physical=65536, neurons_physical=256, axons_physical=256, synaptic_events=1000,
                            latency_steps=10, cores_physical=1, host_macs=0, lanes_per_tile=8)
        area = _terms(_WITH_LANES, values)["chip_area_mm2"]
        assert "area_per_lane(lanes; " in area.source and "tiles" in area.source

    def test_a_declared_lane_block_still_prices_its_three_formulas(self):
        """Each lane row is in its term's priced part and never among its unpriced names ([U-ENERGY class 2] read by
        the rows' names: the word "plane" of a printed reason holds "lane")."""
        terms = _terms(_WITH_LANES, _census(**self._LANE_CENSUS))
        for name, row in (("chip_area_mm2", "area_per_lane"), ("e2e_latency_s", "t_lane_op"),
                          ("energy_dynamic_mj", "e_lane_op")):
            priced, _names = parts(terms[name].source)
            assert row in priced and not any(named(terms[name].source, r) for r in _LANE_ROWS), name

    @pytest.mark.parametrize("profile", ["generic_estimated_22nm", "isaac_like", "truenorth", "loihi", "odin"])
    def test_a_lane_free_report_is_unchanged_by_the_lane_rows(self, profile):
        physics = get_platform_physics(profile)
        values = _census()
        assert _lane_free(price_absolute(values, apply_overrides(physics, _LANES))) == \
            [t.to_dict() for t in price_absolute(values, physics).terms]
