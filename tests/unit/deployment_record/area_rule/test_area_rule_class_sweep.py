"""[U-AREA] The class swept: every reader of chip area in the toolchain reads the area rule on a description with a
block (one test per site; the build record's sweep lists each site with its file:line and this file's test).

Each test calls the site as its callers do and checks that the chip area it reads is the rule path's price for the
census and geometry the site passes (``area_rule.pricing.rule_area``), not R5's path. The sites priced inside
``price_absolute`` itself (the pricer, the flags, the external gate, the lanes, the outside part, the converter kinds)
are tested in test_area_rule_pricing.py and test_area_rule_pricer.py.
"""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute.host_split import price_shares
from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record, candidate_cost_report
from mimarsinan.deployment_record.cost.absolute.validity import UNJUDGED, CoreGeometry, judged_geometry
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.area_rule.pricing import rule_area
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.study import compare_platforms

from unit.deployment_record.area_rule import _golden as G

AREA = "chip_area_mm2"
#: A census of 100 cores of 256 x 256 (the cross-platform study's own census, test values).
_CENSUS = {"cores_physical": 100, "cells_physical": 100 * 256 * 256, "neurons_physical": 100 * 256,
           "axons_physical": 100 * 256, "weight_bits": 4, "macs": 1e7, "synaptic_events": 1e7, "latency_steps": 64,
           "timesteps": 32, "sync_count": 8, "segment_cores": 100, "tiles": 25, "host_macs": 0}
_GEOMETRY = (CoreGeometry(256, 256, 100),)
BLOCKS = ("truenorth", "generic_estimated_22nm", "neurram", "hermes_64", "loihi")


def _q(values):
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _rule(name, quantities, geometry):
    physics = get_platform_physics(name)
    return rule_area(physics, quantities, judged_geometry(geometry, quantities)).band[1]


class TestTheReaders:
    @pytest.mark.parametrize("name", BLOCKS)
    def test_the_cross_platform_table_reads_the_rule(self, name):
        """study.py:108 (compare_platforms)."""
        row = compare_platforms(_q(_CENSUS), (name,), geometry=_GEOMETRY).rows[0]
        assert G.close(row.values[AREA], _rule(name, _q(_CENSUS), _GEOMETRY))

    def test_the_table_under_census_proved_refuses_a_pooled_census_by_name(self):
        """study.py:108 without the declared core types (CENSUS_PROVED): the rule's band would rest on a mean core."""
        row = compare_platforms(_q(_CENSUS), ("generic_estimated_22nm",)).rows[0]
        assert AREA not in row.values and "pass the declared core types" in row.unavailable[AREA]

    @pytest.mark.parametrize("name", BLOCKS)
    def test_the_share_variants_read_the_rule(self, name):
        """cost/absolute/host_split.py:171 (price_shares): every variant keeps the chip's area."""
        terms, _refusals = price_shares(_q(_CENSUS), get_platform_physics(name), geometry=_GEOMETRY)
        want = _rule(name, _q(_CENSUS), _GEOMETRY)
        assert all(G.close(variant[AREA].nominal, want) for variant in terms.values())

    @pytest.mark.parametrize("name", BLOCKS)
    def test_the_candidate_report_reads_the_rule(self, name):
        """cost/absolute/report.py:119 (candidate_cost_report, the candidate view's report, objectives/views.py:190;
        its reader is the catalog's priced_term, objectives/catalog.py:197)."""
        report = candidate_cost_report(_q(_CENSUS), get_platform_physics(name), geometry=None)
        area = next(t for t in report.area if t.name == AREA)
        assert G.close(area.value, _rule(name, _q(_CENSUS), UNJUDGED))

    def test_the_candidate_objective_reads_the_rule(self):
        """objectives/views.py:190 and objectives/catalog.py:197: the chip_area_mm2 objective of a candidate."""
        from mimarsinan.deployment_record.objectives.catalog import OBJECTIVES
        from unit.deployment_record.test_fidelity_from_record import _candidate
        candidate = _candidate()
        spec = next(s for s in OBJECTIVES.all() if s.key == AREA)
        area = next(t for t in candidate.cost_report().area if t.name == AREA)
        assert spec.available(candidate) and G.close(spec.value(candidate), area.value)
        assert "area_rule T1 anchor" in area.source

    def test_a_sealed_record_reads_its_sealed_physics(self):
        """cost/absolute/report.py:154 (absolute_pricing_for_record, the record view, objectives/views.py:276): a
        record sealed with a block prices by the rule; the sealed physics decides, never the current profile."""
        from unit.deployment_record.test_fidelity_from_record import _record
        record = _record()
        area = next(t for t in absolute_pricing_for_record(record).terms if t.name == AREA)
        assert area.source.startswith("area_rule T1 anchor (die)")

    def test_the_level_plane_reads_the_rule(self):
        """cost/absolute/level_pricing.py:19 (price_level, the level plane): the level's census at the chip's declared
        core type prices its core grid by the rule, the adders beside it."""
        from mimarsinan.deployment_record.cost.absolute.level_pricing import price_level
        from mimarsinan.deployment_record.cost.absolute.validity import declared_geometry
        from unit.deployment_record.test_level_pricing import _view
        from unit.mapping.capability.ir_fixtures import lane_graph
        view, physics = _view(lane_graph(), "C0", 4), get_platform_physics("generic_estimated_22nm")
        area = next(t for t in price_level(view, physics).terms if t.name == AREA)
        rule = rule_area(physics, view.quantities,
                         judged_geometry(declared_geometry(view.chip["cores"]), view.quantities))
        assert area.source.startswith(rule.basis) and area.value >= rule.band[1]

    @pytest.mark.parametrize("target,basis", [("hermes_64", "area_rule T4 anchor (die)"),
                                              ("generic_estimated_22nm", "area_rule T2_digital type rule")])
    def test_the_ladder_program_reads_the_rule(self, target, basis):
        """introspection/scaling_ladder.py:200 (price_scheduled's program terms; scripts/scaling_ladder.py:423 calls
        it): a ladder point's chip area is the rule path's, from the physics its platform resolves."""
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
        from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
        from unit.deployment_record.introspection import test_study_cost_cell_ue4 as UE4
        network = UE4._network((256, 256, 64))
        platform = build_platform_constraints_resolved(merge_pipeline_config({}, {
            "target": target, "allow_scheduling": False, "target_overrides": {"cores": {"value": [
                {"max_axons": 256, "max_neurons": 64, "count": 5, "has_bias": True}], "note": "test cores"}}}))
        census = UE4.LD.shape_census(network, platform)
        scheduled = UE4.LD.schedule_program(network, census, platform, UE4._ladder().program_facts,
                                            tiles=UE4.LD.packed_tiles(census, platform))
        assert scheduled.refusal is None, scheduled.refusal
        priced = UE4.LD.price_scheduled(network, census, scheduled, platform)
        assert priced.terms["program"][AREA].basis.startswith(basis)

    def test_the_correlation_case_reads_the_rule(self):
        """correlation/run.py:125 (correlate): TrueNorth's die from its core count is the anchor's identity."""
        from mimarsinan.deployment_record.correlation.library import get_case
        from mimarsinan.deployment_record.correlation.run import correlate
        result = correlate(get_case("truenorth_die_area"))
        axis = next(a for a in result.axes if a.name == AREA)
        assert G.close(axis.predicted, G.golden()["T-3/T-4 anchor_identities (lanes out)"]["truenorth"]["nominal"])
