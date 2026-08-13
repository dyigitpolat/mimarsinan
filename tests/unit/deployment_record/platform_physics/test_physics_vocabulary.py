"""The constant vocabulary is a closed, dimensioned declaration — the wizard's layout too."""

import pytest

from mimarsinan.deployment_record.platform_physics.constants import (
    PHYSICS_CONSTANTS,
    PHYSICS_GROUPS,
    keys_in_group,
    spec_for,
)
from mimarsinan.deployment_record.units import DIMENSIONS, unit_for


def test_every_spec_declares_a_known_group_and_dimension():
    for key, spec in PHYSICS_CONSTANTS.items():
        assert spec.key == key, "the mapping key and the spec key must not drift"
        assert spec.group in PHYSICS_GROUPS
        assert spec.dimension in DIMENSIONS


def test_every_spec_display_unit_matches_its_own_dimension():
    for spec in PHYSICS_CONSTANTS.values():
        assert unit_for(spec.display_unit).dimension == spec.dimension


def test_every_spec_states_what_it_is_per():
    """A bare 'energy' constant is unusable: the doc must name the multiplicand."""
    for spec in PHYSICS_CONSTANTS.values():
        assert len(spec.doc.strip()) >= 20, spec.key
        assert spec.multiplicand.strip(), spec.key


def test_groups_partition_the_vocabulary():
    covered = [key for group in PHYSICS_GROUPS for key in keys_in_group(group)]
    assert sorted(covered) == sorted(PHYSICS_CONSTANTS)
    assert len(covered) == len(set(covered)), "a constant may belong to exactly one group"


def test_the_vocabulary_covers_each_absolute_objective_it_must_back():
    """Area, energy, latency and throughput each need at least one declared constant."""
    dimensions = {spec.dimension for spec in PHYSICS_CONSTANTS.values()}
    assert {"area", "energy", "time", "power"} <= dimensions


def test_t_cycle_is_declared_because_it_alone_converts_timesteps_to_seconds():
    spec = spec_for("t_cycle")
    assert spec.dimension == "time"
    assert spec.group == "global"


def test_host_group_exists_so_subsumed_work_is_never_free():
    """Moving work host-side must cost something, or `subsume` wins by disappearing."""
    assert "host" in PHYSICS_GROUPS
    assert keys_in_group("host")


def test_unknown_key_raises_naming_the_vocabulary():
    with pytest.raises(KeyError, match="e_made_up"):
        spec_for("e_made_up")


def test_no_execution_policy_leaked_into_the_physics_vocabulary():
    """Physics is banded numbers only; policy/capability lives on the platform surface."""
    policy_shaped = {"overlap_policy", "weights_resident_across_batch", "schedule_policy"}
    assert not policy_shaped & set(PHYSICS_CONSTANTS)


def test_programming_time_is_declared_per_byte_not_as_a_bandwidth():
    """A multiplicative constant keeps the band monotone; a divisor needs corner flipping."""
    assert "t_program_per_byte" in PHYSICS_CONSTANTS
    assert not any("bandwidth" in key for key in PHYSICS_CONSTANTS)
