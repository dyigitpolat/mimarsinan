"""The shipped profiles: discovered as data, and evidenced by construction."""

import re

import pytest

from mimarsinan.deployment_record.platform_physics.constants import PHYSICS_CONSTANTS
from mimarsinan.deployment_record.platform_physics.registry import (
    available_profiles,
    get_platform_physics,
    profile_description_path,
    profiles_dir,
)

_PROFILES = available_profiles()


def test_at_least_one_profile_ships():
    assert _PROFILES


def test_unknown_profile_raises_naming_the_known_ones():
    with pytest.raises(KeyError) as excinfo:
        get_platform_physics("nosuchchip")
    for name in _PROFILES:
        assert name in str(excinfo.value)


@pytest.mark.parametrize("name", _PROFILES)
def test_every_shipped_profile_loads(name):
    physics = get_platform_physics(name)
    assert physics.name == name
    assert physics.constants


@pytest.mark.parametrize("name", _PROFILES)
def test_every_shipped_profile_has_a_description_file_with_content(name):
    path = profile_description_path(name)
    assert path.is_file(), path
    assert len(path.read_text().strip()) > 200, "the description file must carry the research"


@pytest.mark.parametrize("name", _PROFILES)
def test_every_constant_of_every_shipped_profile_carries_its_evidence(name):
    """Research-first is structural: the dataclass refuses unevidenced values, and this
    asserts the shipped data actually exercises that rather than sitting empty."""
    physics = get_platform_physics(name)
    for key, value in physics.constants.items():
        assert key in PHYSICS_CONSTANTS
        assert value.evidence_kind in ("published", "datasheet", "derived", "estimated")


@pytest.mark.parametrize("name", _PROFILES)
def test_every_estimated_constant_is_named_in_the_description_file(name):
    """The owner's rule: an estimate is allowed only with a note in the description file."""
    physics = get_platform_physics(name)
    prose = profile_description_path(name).read_text()
    for key, value in physics.constants.items():
        if value.evidence_kind == "estimated":
            assert re.search(rf"\b{re.escape(key)}\b", prose), (
                f"{name}: estimated constant {key} is not discussed in {name}.md"
            )


@pytest.mark.parametrize("name", _PROFILES)
def test_every_shipped_profile_declares_its_validity_domain(name):
    validity = get_platform_physics(name).validity
    assert validity.measurement_kind
    assert validity.technology_node_nm is not None, "the node the numbers hold at"


def test_profiles_live_beside_the_package_so_the_cwd_never_matters():
    assert profiles_dir().is_dir()
    assert (profiles_dir() / f"{_PROFILES[0]}.json").is_file()


def test_a_profile_is_added_as_data_with_no_source_edit(tmp_path):
    """A vendor drops in two files; the registry is a directory listing, not a table."""
    import json

    from mimarsinan.deployment_record.platform_physics.registry import profiles_in

    (tmp_path / "vendorx.md").write_text("# VendorX\n")
    (tmp_path / "vendorx.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "name": "vendorx",
                "display_name": "Vendor X",
                "description_file": "vendorx.md",
                "validity": {"measurement_kind": "projection", "technology_node_nm": 7.0},
                "constants": {
                    "t_cycle": {
                        "nominal": 1.0,
                        "unit": "ms",
                        "evidence_kind": "estimated",
                        "note": "assumed 1 kHz tick",
                    }
                },
            }
        )
    )
    assert profiles_in(tmp_path) == ("vendorx",)
