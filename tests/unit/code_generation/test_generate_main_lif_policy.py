"""Codegen helpers for orthogonal LIF fire policies."""

from mimarsinan.code_generation.generate_main import resolve_lif_fire_policy


def test_resolve_lif_fire_policy_reset_dimension():
    assert resolve_lif_fire_policy("Default", "<") == "LIFirePolicy<SubtractiveReset, StrictCompare>"
    assert resolve_lif_fire_policy("Novena", "<") == "LIFirePolicy<ZeroReset, StrictCompare>"


def test_resolve_lif_fire_policy_compare_dimension():
    assert resolve_lif_fire_policy("Default", "<=") == "LIFirePolicy<SubtractiveReset, InclusiveCompare>"
    assert resolve_lif_fire_policy("Default", "<") == "LIFirePolicy<SubtractiveReset, StrictCompare>"
