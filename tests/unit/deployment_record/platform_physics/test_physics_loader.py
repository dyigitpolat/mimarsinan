"""Loading a profile: strict, description-file-backed, and unable to fork a number."""

import json

import pytest

from mimarsinan.chip_simulation.sanafe.presets import PRESETS
from mimarsinan.deployment_record.platform_physics.loader import (
    apply_overrides,
    load_profile,
    profile_from_dict,
)

_MINIMAL = {
    "format_version": 1,
    "name": "tester",
    "display_name": "Test Chip",
    "description_file": "tester.md",
    "validity": {"measurement_kind": "silicon", "technology_node_nm": 14.0},
    "constants": {
        "e_mac": {
            "nominal": 23.6,
            "unit": "pJ",
            "evidence_kind": "published",
            "citation": "davies2018loihi Table 1",
        }
    },
}


def _write(tmp_path, payload, *, with_description=True, name="tester"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    if with_description:
        (tmp_path / payload["description_file"]).write_text("# notes\n")
    return path


def test_a_point_declaration_fills_the_whole_band(tmp_path):
    physics = load_profile(_write(tmp_path, _MINIMAL))
    band = physics.band("e_mac")
    assert (band.low, band.nominal, band.high) == pytest.approx((23.6e-12,) * 3)


def test_the_description_file_must_exist_beside_the_profile(tmp_path):
    path = _write(tmp_path, _MINIMAL, with_description=False)
    with pytest.raises(FileNotFoundError, match="tester.md"):
        load_profile(path)


def test_a_constant_outside_the_vocabulary_is_refused_by_name(tmp_path):
    payload = json.loads(json.dumps(_MINIMAL))
    payload["constants"]["e_hand_waving"] = dict(payload["constants"]["e_mac"])
    with pytest.raises(KeyError, match="e_hand_waving"):
        load_profile(_write(tmp_path, payload))


def test_a_wrong_format_version_is_refused(tmp_path):
    payload = json.loads(json.dumps(_MINIMAL))
    payload["format_version"] = 99
    with pytest.raises(ValueError, match="format_version"):
        load_profile(_write(tmp_path, payload))


def test_the_file_name_must_match_the_declared_profile_name(tmp_path):
    with pytest.raises(ValueError, match="elsewhere"):
        load_profile(_write(tmp_path, _MINIMAL, name="elsewhere"))


def test_a_source_ref_resolves_against_the_sanafe_preset_instead_of_restating_it():
    """The per-event numbers already have a home; a profile REFERENCES it, never copies."""
    payload = json.loads(json.dumps(_MINIMAL))
    payload["constants"]["e_mac"] = {
        "source_ref": "sanafe_preset:loihi:synapse_energy_j",
        "citation": "davies2018loihi",
    }
    physics = profile_from_dict(payload)
    assert physics.band("e_mac").nominal == PRESETS["loihi"]["synapse_energy_j"]
    assert physics.constants["e_mac"].evidence_kind == "derived"
    assert "sanafe_preset:loihi:synapse_energy_j" in physics.constants["e_mac"].derivation


def test_a_source_ref_to_an_unknown_preset_raises_naming_the_known_ones():
    payload = json.loads(json.dumps(_MINIMAL))
    payload["constants"]["e_mac"] = {"source_ref": "sanafe_preset:nosuchchip:synapse_energy_j"}
    with pytest.raises(KeyError, match="nosuchchip"):
        profile_from_dict(payload)


def test_a_source_ref_to_an_unknown_preset_field_raises():
    payload = json.loads(json.dumps(_MINIMAL))
    payload["constants"]["e_mac"] = {"source_ref": "sanafe_preset:loihi:no_such_field"}
    with pytest.raises(KeyError, match="no_such_field"):
        profile_from_dict(payload)


def test_a_source_ref_of_the_wrong_dimension_raises_naming_the_ref():
    """`synapse_latency_s` is a time; binding it to an energy constant must fail loud,
    and the message must name the reference that did it, not just the resolved unit."""
    payload = json.loads(json.dumps(_MINIMAL))
    payload["constants"]["e_mac"] = {"source_ref": "sanafe_preset:loihi:synapse_latency_s"}
    with pytest.raises(ValueError, match="dimension") as excinfo:
        profile_from_dict(payload)
    assert "sanafe_preset:loihi:synapse_latency_s" in str(excinfo.value)


def test_a_source_ref_to_a_field_with_no_unit_suffix_raises():
    """The preset's suffix IS the unit; a field without one has no knowable dimension."""
    payload = json.loads(json.dumps(_MINIMAL))
    payload["constants"]["e_mac"] = {"source_ref": "sanafe_preset:loihi:synapse"}
    with pytest.raises(KeyError, match="synapse"):
        profile_from_dict(payload)


def test_a_constant_may_not_declare_both_a_band_and_a_source_ref():
    payload = json.loads(json.dumps(_MINIMAL))
    payload["constants"]["e_mac"]["source_ref"] = "sanafe_preset:loihi:synapse_energy_j"
    with pytest.raises(ValueError, match="exactly one"):
        profile_from_dict(payload)


def test_a_constant_must_declare_something():
    payload = json.loads(json.dumps(_MINIMAL))
    payload["constants"]["e_mac"] = {"citation": "x"}
    with pytest.raises(ValueError, match="exactly one"):
        profile_from_dict(payload)


def test_overrides_replace_a_value_and_mark_it_as_deviating():
    physics = profile_from_dict(_MINIMAL)
    tuned = apply_overrides(physics, {"e_mac": {"nominal": 30.0, "note": "our silicon"}})
    assert tuned.band("e_mac").nominal == pytest.approx(30e-12)
    assert tuned.constants["e_mac"].overridden
    assert not physics.constants["e_mac"].overridden, "the profile itself is untouched"


def test_an_override_inherits_the_units_of_the_constant_it_replaces():
    physics = profile_from_dict(_MINIMAL)
    tuned = apply_overrides(physics, {"e_mac": {"nominal": 30.0, "note": "n"}})
    assert tuned.constants["e_mac"].unit == "pJ"


def test_an_override_may_add_a_constant_the_profile_never_declared():
    physics = profile_from_dict(_MINIMAL)
    assert not physics.has("t_cycle")
    tuned = apply_overrides(physics, {"t_cycle": {"nominal": 1.0, "unit": "ms", "note": "1 kHz"}})
    assert tuned.band("t_cycle").nominal == pytest.approx(1e-3)


def test_an_override_must_say_why_it_deviates():
    physics = profile_from_dict(_MINIMAL)
    with pytest.raises(ValueError, match="note"):
        apply_overrides(physics, {"e_mac": {"nominal": 30.0}})


def test_an_override_of_an_unknown_constant_raises():
    physics = profile_from_dict(_MINIMAL)
    with pytest.raises(KeyError, match="e_nope"):
        apply_overrides(physics, {"e_nope": {"nominal": 1.0, "note": "n"}})


def test_empty_overrides_return_the_same_profile():
    physics = profile_from_dict(_MINIMAL)
    assert apply_overrides(physics, {}) == physics


def test_overrides_are_recorded_so_a_run_states_the_physics_it_used():
    physics = profile_from_dict(_MINIMAL)
    tuned = apply_overrides(physics, {"e_mac": {"nominal": 30.0, "note": "our silicon"}})
    assert tuned.overridden_keys() == ("e_mac",)
    assert physics.overridden_keys() == ()
