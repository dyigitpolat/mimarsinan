"""A published aggregate already contains its decomposition — charge one, never both."""

import pytest

from mimarsinan.deployment_record.platform_physics.constants import (
    PHYSICS_CONSTANTS,
    SUPERSEDES,
    resolve_supersessions,
    superseded_by,
)
from mimarsinan.deployment_record.platform_physics.registry import get_platform_physics


def test_every_key_named_in_a_supersede_rule_is_in_the_vocabulary():
    for aggregate, members in SUPERSEDES.items():
        assert aggregate in PHYSICS_CONSTANTS
        for member in members:
            assert member in PHYSICS_CONSTANTS, f"{aggregate} supersedes unknown {member}"


def test_an_aggregate_never_supersedes_itself_or_another_aggregate():
    for aggregate, members in SUPERSEDES.items():
        assert aggregate not in members
        assert not set(members) & set(SUPERSEDES)


def test_a_superseded_term_drops_out_of_the_priceable_set():
    survivors = resolve_supersessions(("e_synaptic_event_total", "e_mac", "t_cycle"))
    assert survivors == ("e_synaptic_event_total", "t_cycle")


def test_without_the_aggregate_the_decomposition_stands():
    assert resolve_supersessions(("e_mac", "e_inter_tile_hop")) == (
        "e_inter_tile_hop",
        "e_mac",
    )


def test_supersession_is_independent_of_declaration_order():
    forward = resolve_supersessions(("e_synaptic_event_total", "p_static_per_core"))
    backward = resolve_supersessions(("p_static_per_core", "e_synaptic_event_total"))
    assert forward == backward == ("e_synaptic_event_total",)


def test_superseded_by_is_empty_for_an_ordinary_constant():
    assert superseded_by("t_cycle") == ()


def test_the_truenorth_profile_exercises_the_rule():
    """Its 26 pJ/event is whole-chip power over events; the static floor is inside it."""
    physics = get_platform_physics("truenorth")
    assert physics.has("e_synaptic_event_total")
    assert physics.has("p_static_per_core")
    priceable = resolve_supersessions(physics.constants)
    assert "e_synaptic_event_total" in priceable
    assert "p_static_per_core" not in priceable
    assert "e_inter_tile_hop" not in priceable


def test_the_aggregates_own_doc_warns_against_summing_both():
    doc = PHYSICS_CONSTANTS["e_synaptic_event_total"].doc
    assert "SUPERSEDES" in doc


@pytest.mark.parametrize("key", sorted(SUPERSEDES))
def test_an_aggregate_lives_in_the_aggregate_group(key):
    assert PHYSICS_CONSTANTS[key].group == "aggregate"
