"""The reference case: a published measurement plus the census that produced it."""

import pytest

from mimarsinan.deployment_record.correlation.case import (
    PublishedValue,
    ReferenceCase,
)

_CENSUS = {"cores_physical": 1.0, "synaptic_events": 1e6}
_PUBLISHED = {"average_power_mw": PublishedValue(
    value=477.0, citation="frenkel2019odin IV-A", quote="P of ODIN is 477uW")}


def _case(**over):
    kwargs = dict(
        name="a_case", profile="odin", citation="frenkel2019odin",
        description="d", independence="independent",
        operating_conditions={"supply_v": 0.55},
        census=_CENSUS, census_sources={k: "published" for k in _CENSUS},
        overrides={}, published=_PUBLISHED, tolerance_pct=25.0,
    )
    kwargs.update(over)
    return ReferenceCase(**kwargs)


class TestTheCaseIsSelfDescribing:
    def test_it_carries_its_published_value_with_a_citation(self):
        case = _case()
        assert case.published["average_power_mw"].citation

    def test_every_census_number_states_where_it_came_from(self):
        """A census entry without a source is a fitted number wearing a disguise."""
        with pytest.raises(ValueError, match="census_sources"):
            _case(census_sources={"cores_physical": "published"})

    def test_a_census_key_outside_the_quantity_catalog_fails_loud(self):
        with pytest.raises(KeyError, match="not a catalog quantity"):
            _case(census={"invented_quantity": 1.0},
                  census_sources={"invented_quantity": "published"})

    def test_independence_must_be_declared_from_the_closed_set(self):
        with pytest.raises(ValueError, match="independence"):
            _case(independence="probably fine")

    def test_a_self_consistency_case_says_so(self):
        """A case whose published value produced the constant is NOT evidence the
        model generalizes, and the artifact has to carry that distinction."""
        case = _case(independence="self_consistency")
        assert case.is_independent is False
        assert _case().is_independent is True


class TestRefusals:
    def test_a_case_publishing_nothing_fails_loud(self):
        with pytest.raises(ValueError, match="at least one measured value"):
            _case(published={})

    def test_a_nonpositive_tolerance_fails_loud(self):
        with pytest.raises(ValueError, match="tolerance"):
            _case(tolerance_pct=0.0)

    def test_a_published_value_needs_its_quote(self):
        with pytest.raises(ValueError, match="quote"):
            _case(published={"average_power_mw": PublishedValue(
                value=1.0, citation="c", quote="")})


class TestTheArtifact:
    def test_it_round_trips(self):
        case = _case()
        assert ReferenceCase.from_dict(case.to_dict()) == case

    def test_an_unknown_field_is_rejected(self):
        payload = _case().to_dict()
        payload["surprise"] = 1
        with pytest.raises(ValueError, match="unknown fields"):
            ReferenceCase.from_dict(payload)
