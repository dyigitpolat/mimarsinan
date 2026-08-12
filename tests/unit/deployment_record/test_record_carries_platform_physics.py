"""The sealed record must state the physics its numbers were priced with."""

import json

from mimarsinan.deployment_record.schema.record import RecordIdentity
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)


def _identity(config):
    return RecordIdentity(
        format_version=1,
        run_dir="/tmp/run",
        cell_key="t0_test",
        mode="lif",
        model_type="simple_mlp",
        model_name="simple_mlp",
        workload="mnist",
        config_digest="deadbeef",
        platform=build_platform_constraints_resolved(config),
        deployment_options={},
        created_at="2026-08-13T00:00:00",
    )


def test_a_run_with_no_declared_profile_records_no_physics():
    """Absence must survive into the record: no profile means no absolute number."""
    assert _identity({}).platform["platform_physics_resolved"] is None


def test_the_record_carries_the_resolved_constants_not_just_the_profile_name():
    """A later edit to the profile file must not silently rewrite what priced this run."""
    identity = _identity({"platform_physics_profile": "truenorth"})
    physics = identity.platform["platform_physics_resolved"]
    assert physics["name"] == "truenorth"
    assert physics["constants"]["e_synaptic_event_total"]["nominal"] == 26.0
    assert physics["constants"]["e_synaptic_event_total"]["citation"]
    assert physics["validity"]["technology_node_nm"] == 28.0


def test_operator_overrides_are_visible_in_the_record_as_deviations():
    identity = _identity({
        "platform_physics_profile": "truenorth",
        "platform_physics_overrides": {
            "t_cycle": {"nominal": 500.0, "unit": "us", "note": "overclocked part"}
        },
    })
    constant = identity.platform["platform_physics_resolved"]["constants"]["t_cycle"]
    assert constant["overridden"] is True
    assert constant["note"] == "overclocked part"


def test_the_physics_survives_the_records_json_round_trip():
    identity = _identity({"platform_physics_profile": "truenorth"})
    restored = RecordIdentity.from_dict(json.loads(json.dumps(identity.to_dict())))
    assert (
        restored.platform["platform_physics_resolved"]
        == identity.platform["platform_physics_resolved"]
    )
