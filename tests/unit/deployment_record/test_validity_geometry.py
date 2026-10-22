"""E190: aggregate constants hold at their measured core and refuse by name outside it."""

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute, validity_refusals
from mimarsinan.deployment_record.cost.absolute.validity import CENSUS_PROVED, UNJUDGED, CoreGeometry, geometry_of
from mimarsinan.deployment_record.objectives.views import CandidateStaticView
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.platform_physics.vocabulary import SUPERSEDES
from mimarsinan.deployment_record.platform_physics.profile import (
    PlatformPhysics,
    PlatformPhysicsValidity,
    ValidityGeometry,
)
from mimarsinan.deployment_record.quantities.from_candidate import CandidateQuantityContext
from mimarsinan.deployment_record.study import compare_platforms

from unit.deployment_record.g3_fixtures import quantities


def _area(physics, geometry, *, cores, bits=8):
    pricing = price_absolute(quantities(cores_physical=cores, host_macs=0, weight_bits=bits), physics,
                             geometry=geometry)
    terms = {t.name: t for t in pricing.terms}
    refusals = {r.name: r.reason for r in pricing.refusals}
    return terms.get("chip_area_mm2"), refusals.get("chip_area_mm2")


def _chip(axons, neurons, count):
    return (CoreGeometry(axons, neurons, count),)


class TestShippedProfiles:
    def test_loihi_on_a_256_by_256_core_refuses_the_area_by_name(self):
        term, reason = _area(get_platform_physics("loihi"), _chip(256, 256, 16), cores=16)
        assert term is None
        assert "area_per_core_total lies outside its validity" in reason and "128 x 1024" in reason

    def test_truenorth_at_its_own_core_is_unchanged(self):
        truenorth = get_platform_physics("truenorth")
        judged, reason = _area(truenorth, _chip(256, 256, 20), cores=20)
        unjudged, _ = _area(truenorth, None, cores=20)
        assert reason is None
        assert judged.to_dict() == unjudged.to_dict()

    def test_the_ex_j_fixture_refuses(self):
        """115 cores priced at 0.46875 mm^2 each on a searched geometry: refused, never 53.9 mm^2."""
        term, reason = _area(get_platform_physics("loihi"), _chip(256, 512, 115), cores=115)
        assert term is None and reason

    def test_wider_weights_than_the_cell_refuse(self):
        term, reason = _area(get_platform_physics("odin"), _chip(256, 256, 1), cores=1, bits=8)
        assert term is None and "3-bit cells" in reason
        term, reason = _area(get_platform_physics("odin"), _chip(256, 256, 1), cores=1, bits=3)
        assert reason is None and term is not None

    def test_a_chip_beyond_the_core_count_refuses(self):
        term, reason = _area(get_platform_physics("odin"), _chip(256, 256, 2), cores=2, bits=3)
        assert term is None and "chips of 1-1 cores" in reason

    def test_decomposed_constants_price_at_any_geometry(self):
        physics = get_platform_physics("generic_estimated_22nm")
        values = quantities(cores_physical=4, cells_physical=4 * 784 * 512, host_macs=0, weight_bits=8)
        pricing = price_absolute(values, physics, geometry=_chip(784, 512, 4))
        assert "chip_area_mm2" in {t.name for t in pricing.terms}

    @pytest.mark.parametrize("name", ["generic_estimated_22nm", "isaac_like", "loihi", "odin", "truenorth",
                                      "hermes_64", "sram_cim_macro", "neurram"])
    def test_every_shipped_profile_states_a_structured_validity(self, name):
        assert get_platform_physics(name).validity.geometry is not None


def _footprint_profile(scaling):
    """A one-constant footprint profile at a 256 x 256 point, with or without a scaling rule."""
    return profile_from_dict({
        "format_version": 1, "name": "s", "display_name": "S", "description_file": "s.md",
        "validity": {"measurement_kind": "silicon", "geometry": {
            "axons": 256, "neurons": 256, "cell_bits": None, "core_count": None, "scaling": scaling}},
        "constants": {"area_per_core_total": {"nominal": 100000.0, "unit": "um^2",
                                              "evidence_kind": "estimated", "note": "test"}},
    })


class TestTheRule:
    def _scaled(self, scaling):
        return _footprint_profile(scaling)

    def test_a_declared_scaling_rule_prices_outside_its_point(self):
        term, reason = _area(self._scaled("cells"), _chip(512, 256, 3), cores=3)
        assert reason is None
        assert term.value == pytest.approx(3 * 2 * 0.1)
        assert "scaled by core cells" in term.source

    def test_without_a_scaling_rule_the_same_chip_refuses(self):
        term, reason = _area(self._scaled(None), _chip(512, 256, 3), cores=3)
        assert term is None and "no scaling rule" in reason

    def test_validity_refusals_name_every_declared_aggregate(self):
        cores = [{"max_axons": 784, "max_neurons": 512, "count": 60}]
        refusals = validity_refusals(get_platform_physics("truenorth"), cores, weight_bits=5)
        assert [r.name for r in refusals] == ["area_per_core_total"]
        assert validity_refusals(get_platform_physics("truenorth"),
                                 [{"max_axons": 256, "max_neurons": 256, "count": 1}], weight_bits=5) == ()

    def test_physics_without_a_structured_validity_is_read_as_declared(self):
        physics = self._scaled(None)
        bare = PlatformPhysics.from_dict({**physics.to_dict(), "validity": {"measurement_kind": "silicon"}})
        term, reason = _area(bare, _chip(512, 256, 3), cores=3)
        assert reason is None and term.value == pytest.approx(0.3)

    def test_a_validity_without_geometry_serializes_as_before(self):
        validity = PlatformPhysicsValidity(measurement_kind="silicon")
        assert "geometry" not in validity.to_dict()
        geometry = ValidityGeometry(axons=1, neurons=2, core_count=(1, 4))
        assert ValidityGeometry.from_dict(geometry.to_dict()) == geometry

    def test_geometry_reads_the_declared_core_types(self):
        assert geometry_of([{"max_axons": 3, "max_neurons": 4}]) == (CoreGeometry(3, 4, 1),)


def test_the_candidate_view_judges_its_own_declared_cores():
    """A searched candidate off TrueNorth's core: the live view refuses the area by name."""
    context = CandidateQuantityContext(cores_physical=64, host_macs=0, weight_bits=4,
                                       core_geometry=_chip(160, 160, 64))
    view = CandidateStaticView(layout=None, chip_param_capacity=None, total_params=None,
                               host_side_segment_count=None, physics=get_platform_physics("truenorth"),
                               quantity_context=context)
    notes = view.cost_report().notes
    assert any(n.startswith("refused chip_area_mm2: area_per_core_total lies outside") for n in notes)


def test_the_cross_platform_study_judges_the_stated_geometry():
    census = quantities(cores_physical=4, host_macs=0, weight_bits=4)
    table = compare_platforms(census, ["truenorth"], geometry=_chip(160, 160, 4))
    assert "lies outside its validity" in table.rows[0].unavailable["chip_area_mm2"]
    assert "chip_area_mm2" in compare_platforms(census, ["truenorth"]).rows[0].values


def _per_event_physics():
    return profile_from_dict({
        "format_version": 1, "name": "e", "display_name": "E", "description_file": "e.md",
        "validity": {"measurement_kind": "silicon", "geometry": {
            "axons": 256, "neurons": 256, "cell_bits": None, "core_count": None, "scaling": "cells"}},
        "constants": {"e_synaptic_event_total": {"nominal": 10.0, "unit": "pJ", "evidence_kind": "estimated",
                                                 "note": "test"}},
    })


class TestThePerEventAggregate:
    def test_it_refuses_the_energy_by_name_off_its_core_even_with_a_scaling_rule(self):
        values = quantities(synaptic_events=1000, host_macs=0, cores_physical=1)
        off = price_absolute(values, _per_event_physics(), geometry=_chip(128, 128, 1))
        assert "energy_per_inference_mj" not in {t.name for t in off.terms}
        reason = {r.name: r.reason for r in off.refusals}["energy_per_inference_mj"]
        assert reason.startswith("e_synaptic_event_total lies outside its validity")
        on = price_absolute(values, _per_event_physics(), geometry=_chip(256, 256, 1))
        assert "energy_per_inference_mj" in {t.name for t in on.terms}

    def test_it_absorbs_the_value_noc_row(self):
        assert "e_noc_value_per_byte_hop" in SUPERSEDES["e_synaptic_event_total"]


class TestAnOmittedGeometry:
    """A caller cannot skip the validity by omission: the census's own core geometry is judged."""

    _CENSUS = dict(cores_physical=16, axons_physical=16 * 256, neurons_physical=16 * 256,
                   cells_physical=16 * 256 * 256, host_macs=0)

    def test_the_census_geometry_is_judged_when_the_caller_states_none(self):
        pricing = price_absolute(quantities(**self._CENSUS), get_platform_physics("loihi"))
        reason = {r.name: r.reason for r in pricing.refusals}["chip_area_mm2"]
        assert "lies outside its validity" in reason and "256 x 256" in reason

    def test_unjudged_names_the_choice_at_the_call_site(self):
        pricing = price_absolute(quantities(**self._CENSUS), get_platform_physics("loihi"), geometry=UNJUDGED)
        assert "chip_area_mm2" in {t.name for t in pricing.terms}

    def test_the_cross_platform_study_judges_the_census_geometry(self):
        row = compare_platforms(quantities(**self._CENSUS), ["loihi"]).rows[0]
        assert "lies outside its validity" in row.unavailable["chip_area_mm2"]


class TestACensusThatPoolsCoreTypes:
    """A census proves one core type only when its cells equal cores x mean axons x mean neurons;
    a pooled census names no core, so every aggregate with a validity point refuses by name."""

    def test_two_core_types_never_read_as_their_average(self):
        """2 x 384 x 384 + 2 x 128 x 128 is not 4 x 256 x 256: TrueNorth's area refuses, as it does declared."""
        census = quantities(cores_physical=4, axons_physical=2 * 384 + 2 * 128, neurons_physical=2 * 384 + 2 * 128,
                            cells_physical=2 * 384 * 384 + 2 * 128 * 128, host_macs=0, weight_bits=8)
        derived = {r.name: r.reason for r in price_absolute(census, get_platform_physics("truenorth")).refusals}
        assert "several core types" in derived["chip_area_mm2"]
        assert "327680" in derived["chip_area_mm2"] and "262144" in derived["chip_area_mm2"]
        declared = price_absolute(census, get_platform_physics("truenorth"),
                                  geometry=geometry_of([{"max_axons": 384, "max_neurons": 384, "count": 2},
                                                        {"max_axons": 128, "max_neurons": 128, "count": 2}]))
        assert "chip_area_mm2" in {r.name for r in declared.refusals}

    def test_the_t0_41_chip_names_no_core_it_lacks(self):
        census = quantities(cores_physical=120, axons_physical=60 * 784 + 60 * 512,
                            neurons_physical=60 * 512 + 60 * 256,
                            cells_physical=60 * 784 * 512 + 60 * 512 * 256, host_macs=0, weight_bits=8)
        reason = {r.name: r.reason for r in price_absolute(census, get_platform_physics("loihi")).refusals}[
            "chip_area_mm2"]
        assert "several core types" in reason and "648" not in reason

    def test_a_census_of_fractional_cores_pools_types(self):
        census = quantities(cores_physical=3, axons_physical=784 + 512 + 512, neurons_physical=3 * 256,
                            cells_physical=784 * 256 + 2 * 512 * 256, host_macs=0)
        reason = {r.name: r.reason for r in price_absolute(census, get_platform_physics("truenorth")).refusals}[
            "chip_area_mm2"]
        assert "several core types" in reason

    def test_axons_that_are_not_whole_per_core_pool_types_even_where_the_cells_agree(self):
        """16 x 192 + 100 x 384 + 192 x 192: 78336 cells = 3 x floor(308 / 3) x 256, so only the whole-core
        guard sees that these three cores are no 102 x 256 type."""
        census = quantities(cores_physical=3, axons_physical=16 + 100 + 192, neurons_physical=192 + 384 + 192,
                            cells_physical=16 * 192 + 100 * 384 + 192 * 192, host_macs=0)
        assert 16 * 192 + 100 * 384 + 192 * 192 == 3 * (308 // 3) * 256
        reason = {r.name: r.reason for r in price_absolute(census, get_platform_physics("truenorth")).refusals}[
            "chip_area_mm2"]
        assert "several core types" in reason and "not whole per core" in reason

    def test_a_census_without_its_cells_proves_no_one_type(self):
        census = quantities(cores_physical=4, axons_physical=4 * 256, neurons_physical=4 * 256, host_macs=0)
        reason = {r.name: r.reason for r in price_absolute(census, get_platform_physics("truenorth")).refusals}[
            "chip_area_mm2"]
        assert "cells_physical" in reason

    def test_the_residual_pool_the_cells_identity_cannot_see_is_never_judged_silently(self):
        """Targeted pass: the identity is necessary, not sufficient. 384 x 256 + 128 x 256 averages to two
        256 x 256 cores, so a census proves only its cell count. ``price_absolute`` judges it at the mean
        core and says so in the term's basis; the cross-platform table, whose rows carry no source,
        refuses by name (``CENSUS_PROVED``); the declared types refuse; UNJUDGED stays explicit."""
        census = quantities(cores_physical=2, axons_physical=384 + 128, neurons_physical=2 * 256,
                            cells_physical=384 * 256 + 128 * 256, host_macs=0, weight_bits=4)
        assert (384 * 256 + 128 * 256) == 2 * 256 * 256
        row = compare_platforms(census, ["truenorth"]).rows[0]
        assert "chip_area_mm2" not in row.values
        reason = row.unavailable["chip_area_mm2"]
        assert "does not prove one core type" in reason and "cell count" in reason
        assert "pass the declared core types" in reason
        truenorth = get_platform_physics("truenorth")
        stated = {t.name: t for t in price_absolute(census, truenorth).terms}["chip_area_mm2"]
        assert "judged at the census's mean core: 2 cores averaging 256 x 256" in stated.source
        strict = price_absolute(census, truenorth, geometry=CENSUS_PROVED)
        assert "chip_area_mm2" in {r.name for r in strict.refusals}
        declared = price_absolute(census, truenorth, geometry=geometry_of(
            [{"max_axons": 384, "max_neurons": 256, "count": 1}, {"max_axons": 128, "max_neurons": 256, "count": 1}]))
        assert "chip_area_mm2" in {r.name for r in declared.refusals}
        unjudged = {t.name: t for t in price_absolute(census, truenorth, geometry=UNJUDGED).terms}["chip_area_mm2"]
        assert unjudged.value == stated.value and "mean core" not in unjudged.source

    def test_a_census_of_one_type_shape_on_many_cores_states_its_mean_core_basis(self):
        census = quantities(cores_physical=20, axons_physical=20 * 256, neurons_physical=20 * 256,
                            cells_physical=20 * 256 * 256, host_macs=0, weight_bits=4)
        truenorth = get_platform_physics("truenorth")
        stated = {t.name: t for t in price_absolute(census, truenorth).terms}["chip_area_mm2"]
        declared = {t.name: t for t in price_absolute(census, truenorth, geometry=_chip(256, 256, 20)).terms}
        assert stated.value == declared["chip_area_mm2"].value
        assert "mean core" in stated.source and "mean core" not in declared["chip_area_mm2"].source
        reason = compare_platforms(census, ["truenorth"]).rows[0].unavailable["chip_area_mm2"]
        assert "pass the declared core types" in reason
        assert "chip_area_mm2" in compare_platforms(census, ["truenorth"], geometry=_chip(256, 256, 20)).rows[0].values

    def test_a_census_of_one_core_proves_its_core_type(self):
        truenorth = get_platform_physics("truenorth")
        at = quantities(cores_physical=1, axons_physical=256, neurons_physical=256, cells_physical=256 * 256,
                        host_macs=0, weight_bits=4)
        off = quantities(cores_physical=1, axons_physical=384, neurons_physical=256, cells_physical=384 * 256,
                         host_macs=0, weight_bits=4)
        area = {t.name: t for t in price_absolute(at, truenorth, geometry=CENSUS_PROVED).terms}["chip_area_mm2"]
        assert "mean core" not in area.source
        reason = {r.name: r.reason for r in price_absolute(off, truenorth).refusals}["chip_area_mm2"]
        assert "384 x 256" in reason and "no scaling rule" in reason


class TestACellScalingRuleReadsTheCensusCells:
    """A declared 'cells' scaling rule prices a footprint by cells, which a census proves exactly."""

    def _cells_profile(self):
        return _footprint_profile("cells")

    def test_a_census_at_the_point_prices_as_before(self):
        census = quantities(cores_physical=2, axons_physical=384 + 128, neurons_physical=2 * 256,
                            cells_physical=2 * 256 * 256, host_macs=0)
        term = {t.name: t for t in price_absolute(census, self._cells_profile()).terms}["chip_area_mm2"]
        unjudged = {t.name: t for t in price_absolute(census, self._cells_profile(), geometry=UNJUDGED).terms}
        assert term.to_dict() == unjudged["chip_area_mm2"].to_dict()

    def test_a_pooled_census_prices_its_cells(self):
        """784 x 512 + 512 x 256 (the t0_41 chip's two types) at 0.1 mm^2 per 256 x 256 footprint."""
        census = quantities(cores_physical=2, axons_physical=784 + 512, neurons_physical=512 + 256,
                            cells_physical=784 * 512 + 512 * 256, host_macs=0)
        term = {t.name: t for t in price_absolute(census, self._cells_profile()).terms}["chip_area_mm2"]
        assert term.value == pytest.approx(0.1 * (784 * 512 + 512 * 256) / (256 * 256))
        declared = {t.name: t for t in price_absolute(census, self._cells_profile(), geometry=geometry_of(
            [{"max_axons": 784, "max_neurons": 512}, {"max_axons": 512, "max_neurons": 256}])).terms}
        assert term.value == pytest.approx(declared["chip_area_mm2"].value)

    def test_a_per_event_aggregate_states_its_mean_core_basis_or_refuses(self):
        """A per-event aggregate is a point model under any rule: a census at the point is judged at its
        mean core with that basis stated, refused under CENSUS_PROVED, and a pooled census refuses."""
        census = quantities(cores_physical=2, axons_physical=2 * 256, neurons_physical=2 * 256,
                            cells_physical=2 * 256 * 256, synaptic_events=1000, host_macs=0)
        stated = {t.name: t for t in price_absolute(census, _per_event_physics()).terms}["energy_per_inference_mj"]
        assert "judged at the census's mean core" in stated.source
        strict = price_absolute(census, _per_event_physics(), geometry=CENSUS_PROVED)
        reason = {r.name: r.reason for r in strict.refusals}["energy_per_inference_mj"]
        assert "point model" in reason and "pass the declared core types" in reason
        pooled = quantities(cores_physical=2, axons_physical=384 + 128, neurons_physical=512 + 256,
                            cells_physical=384 * 512 + 128 * 256, synaptic_events=1000, host_macs=0)
        assert "energy_per_inference_mj" in {r.name for r in price_absolute(pooled, _per_event_physics()).refusals}
        declared = price_absolute(census, _per_event_physics(), geometry=_chip(256, 256, 2))
        energy = {t.name: t for t in declared.terms}["energy_per_inference_mj"]
        assert energy.value == stated.value and "mean core" not in energy.source

    def test_a_per_event_aggregate_is_judged_at_the_mean_core_never_through_the_cells_rule(self):
        """Two cores averaging 128 x 512 hold exactly two 256 x 256 cores' cells, so the cells rule reads
        scale 1; a per-event aggregate is a point model, judged at the mean core, off its point: refused."""
        off = quantities(cores_physical=2, axons_physical=2 * 128, neurons_physical=2 * 512,
                         cells_physical=2 * 128 * 512, synaptic_events=1000, host_macs=0)
        reason = {r.name: r.reason for r in price_absolute(off, _per_event_physics()).refusals}[
            "energy_per_inference_mj"]
        assert "point model" in reason
        declared = price_absolute(off, _per_event_physics(), geometry=_chip(128, 512, 2))
        assert "point model" in {r.name: r.reason for r in declared.refusals}["energy_per_inference_mj"]

    def test_a_pooled_census_at_the_point_cells_still_refuses_a_per_event_aggregate(self):
        """Axons not whole per core pool types, though the stated cells equal two 256 x 256 cores'."""
        pooled = quantities(cores_physical=2, axons_physical=257, neurons_physical=512, cells_physical=2 * 256 * 256,
                            synaptic_events=1000, host_macs=0)
        reason = {r.name: r.reason for r in price_absolute(pooled, _per_event_physics()).refusals}[
            "energy_per_inference_mj"]
        assert "does not prove one core type" in reason and "not whole per core" in reason
