"""Profile + overrides -> the concrete physics a run was priced with."""

import pytest

from mimarsinan.deployment_record.platform_physics.resolve import (
    CUSTOM_PROFILE_NAME,
    resolve_platform_physics,
)


def test_no_profile_and_no_overrides_is_no_physics():
    """A run that declares nothing must report NO area number, not a default one."""
    assert resolve_platform_physics("", {}) is None
    assert resolve_platform_physics(None, None) is None


def test_a_named_profile_resolves_to_its_declared_constants():
    physics = resolve_platform_physics("truenorth", {})
    assert physics is not None
    assert physics.name == "truenorth"
    assert physics.band("t_cycle").nominal == pytest.approx(1e-3)


def test_an_unknown_profile_raises_at_resolution_time():
    """The same loud-at-resolution discipline the floorplan declaration already uses."""
    with pytest.raises(KeyError, match="nosuchchip"):
        resolve_platform_physics("nosuchchip", {})


def test_overrides_apply_on_top_of_the_named_profile():
    physics = resolve_platform_physics(
        "truenorth", {"t_cycle": {"nominal": 500.0, "unit": "us", "note": "overclocked"}}
    )
    assert physics is not None
    assert physics.band("t_cycle").nominal == pytest.approx(5e-4)
    assert physics.overridden_keys() == ("t_cycle",)


def test_overrides_alone_build_a_custom_profile():
    """An operator may price a chip we ship no profile for, by declaring the constants."""
    physics = resolve_platform_physics(
        "", {"t_cycle": {"nominal": 1.0, "unit": "ms", "note": "vendor datasheet"}}
    )
    assert physics is not None
    assert physics.name == CUSTOM_PROFILE_NAME
    assert physics.band("t_cycle").nominal == pytest.approx(1e-3)
    assert physics.overridden_keys() == ("t_cycle",)


def test_a_custom_profile_declares_nothing_it_was_not_given():
    physics = resolve_platform_physics("", {"t_cycle": {"nominal": 1.0, "note": "n"}})
    assert physics is not None
    assert not physics.has("e_mac")


def test_resolution_is_a_pure_function_of_its_inputs():
    """Two identical declarations must resolve identically — the run is reproducible."""
    args = ("truenorth", {"e_mac": {"nominal": 5.0, "note": "n"}})
    assert resolve_platform_physics(*args).to_dict() == resolve_platform_physics(*args).to_dict()


def test_resolving_never_mutates_the_shipped_profile():
    """The registry caches profiles; an override must not leak into the next run."""
    resolve_platform_physics("truenorth", {"t_cycle": {"nominal": 1.0, "note": "n"}})
    again = resolve_platform_physics("truenorth", {})
    assert again is not None
    assert again.overridden_keys() == ()
    assert again.band("t_cycle").nominal == pytest.approx(1e-3)


def test_a_resolved_profile_is_json_safe_so_the_record_can_carry_it():
    import json

    physics = resolve_platform_physics("truenorth", {})
    assert physics is not None
    assert json.loads(json.dumps(physics.to_dict()))["name"] == "truenorth"


def test_an_override_of_an_unknown_constant_raises_at_resolution_time():
    with pytest.raises(KeyError, match="e_nope"):
        resolve_platform_physics("truenorth", {"e_nope": {"nominal": 1.0, "note": "n"}})
