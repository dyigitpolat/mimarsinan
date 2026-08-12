"""A profile value carries evidence, and absence is meaningful — never a default."""

import pytest

from mimarsinan.deployment_record.platform_physics.profile import (
    PhysicsConstantValue,
    PlatformPhysics,
    PlatformPhysicsValidity,
)


def _value(key="e_mac", **over):
    kwargs = dict(
        key=key,
        low=1.0,
        nominal=2.0,
        high=3.0,
        unit="pJ",
        evidence_kind="published",
        citation="davies2018loihi Table 1",
        derivation="",
        note="",
    )
    kwargs.update(over)
    return PhysicsConstantValue(**kwargs)


def _physics(constants=None, name="tester"):
    return PlatformPhysics(
        name=name,
        display_name="Test Chip",
        description_file=f"{name}.md",
        validity=PlatformPhysicsValidity(measurement_kind="silicon"),
        constants={v.key: v for v in (constants or [_value()])},
    )


def test_published_evidence_requires_a_citation():
    with pytest.raises(ValueError, match="citation"):
        _value(evidence_kind="published", citation="")


def test_datasheet_evidence_requires_a_citation():
    with pytest.raises(ValueError, match="citation"):
        _value(evidence_kind="datasheet", citation="")


def test_derived_evidence_requires_the_derivation_shown():
    with pytest.raises(ValueError, match="derivation"):
        _value(evidence_kind="derived", citation="x", derivation="")


def test_estimated_evidence_requires_a_written_rationale():
    """The owner's rule: an estimate is allowed, an UNEXPLAINED estimate is not."""
    with pytest.raises(ValueError, match="note"):
        _value(evidence_kind="estimated", citation="", note="")


def test_estimated_evidence_is_accepted_when_it_explains_itself():
    value = _value(evidence_kind="estimated", citation="", note="scaled from 28nm by node ratio")
    assert value.evidence_kind == "estimated"


def test_unknown_evidence_kind_raises():
    with pytest.raises(ValueError, match="evidence_kind"):
        _value(evidence_kind="vibes")


def test_a_value_for_a_key_outside_the_vocabulary_raises():
    with pytest.raises(KeyError, match="e_not_a_constant"):
        _value(key="e_not_a_constant")


def test_declared_unit_must_match_the_specs_dimension():
    with pytest.raises(ValueError, match="dimension"):
        _value(key="e_mac", unit="ns")


def test_band_must_be_ordered():
    with pytest.raises(ValueError, match="low <= nominal <= high"):
        _value(low=3.0, nominal=2.0, high=1.0)


def test_a_point_value_may_set_all_three_corners():
    value = _value(low=23.6, nominal=23.6, high=23.6)
    assert value.band.low == value.band.high


def test_the_band_is_derived_from_the_declaration_and_cannot_drift():
    """There is one number in the file; the SI band is computed, never stored beside it."""
    value = _value(low=1.0, nominal=23.6, high=100.0, unit="pJ")
    assert value.band.nominal == pytest.approx(23.6e-12)
    assert value.band.low == pytest.approx(1e-12)
    assert value.band.high == pytest.approx(100e-12)
    assert "band" not in {f for f in value.to_dict()}


def test_the_bands_basis_states_the_evidence():
    """Band.basis is required non-empty; for physics it must carry the citation."""
    value = _value(evidence_kind="published", citation="davies2018loihi Table 1")
    assert "published" in value.band.basis
    assert "davies2018loihi Table 1" in value.band.basis


def test_band_returns_canonical_si_and_has_returns_presence():
    physics = _physics()
    assert physics.has("e_mac")
    assert physics.band("e_mac").nominal == pytest.approx(2e-12)


def test_absence_raises_naming_the_constant_and_never_defaults():
    """No get-with-default anywhere: an undeclared constant disables its objectives."""
    physics = _physics()
    assert not physics.has("area_per_cell")
    with pytest.raises(KeyError, match="area_per_cell"):
        physics.band("area_per_cell")
    assert not hasattr(physics, "get")


def test_missing_reports_exactly_the_undeclared_subset():
    physics = _physics()
    assert physics.missing(("e_mac", "area_per_cell", "t_cycle")) == (
        "area_per_cell",
        "t_cycle",
    )
    assert physics.missing(("e_mac",)) == ()


def test_declares_all_is_the_availability_predicate_objectives_will_use():
    physics = _physics()
    assert physics.declares_all(("e_mac",))
    assert not physics.declares_all(("e_mac", "t_cycle"))


def test_evidence_kinds_summarizes_what_a_report_must_disclose():
    physics = _physics(
        [
            _value("e_mac", evidence_kind="published", citation="p"),
            _value("t_cycle", unit="ns", evidence_kind="estimated",
                   citation="", note="assumed 1 kHz tick"),
        ]
    )
    assert physics.evidence_kinds() == {"e_mac": "published", "t_cycle": "estimated"}


def test_a_constants_map_whose_key_disagrees_with_its_value_raises():
    with pytest.raises(ValueError, match="keyed"):
        PlatformPhysics(
            name="tester",
            display_name="Test Chip",
            description_file="tester.md",
            validity=PlatformPhysicsValidity(measurement_kind="silicon"),
            constants={"t_cycle": _value("e_mac")},
        )


def test_profile_round_trips_through_json():
    physics = _physics()
    assert PlatformPhysics.from_dict(physics.to_dict()) == physics


def test_unknown_field_in_a_serialized_profile_raises():
    payload = _physics().to_dict()
    payload["surprise"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        PlatformPhysics.from_dict(payload)


def test_validity_records_whether_numbers_are_silicon_or_projection():
    with pytest.raises(ValueError, match="measurement_kind"):
        PlatformPhysicsValidity(measurement_kind="guessed")
