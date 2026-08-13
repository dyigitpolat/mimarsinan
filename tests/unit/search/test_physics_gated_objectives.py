"""A search may optimize a vendor-priced axis exactly when its target declares one."""

import pytest

from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.search.results import (
    resolve_active_objectives,
    resolve_active_specs,
)

_TRUENORTH = get_platform_physics("truenorth")
_PRICED = "chip_area_mm2"


def test_a_run_declaring_no_physics_is_refused_by_name():
    with pytest.raises(ValueError, match=_PRICED):
        resolve_active_specs("joint", ("estimated_accuracy", _PRICED), physics=None)


def test_the_refusal_states_what_the_axis_requires():
    with pytest.raises(ValueError, match="physics"):
        resolve_active_specs("joint", (_PRICED,), physics=None)


def test_a_run_declaring_physics_may_optimize_it():
    specs = resolve_active_specs(
        "joint", ("estimated_accuracy", _PRICED), physics=_TRUENORTH
    )
    assert [spec.key for spec in specs] == ["estimated_accuracy", _PRICED]


def test_the_projection_carries_the_gate_too():
    objectives = resolve_active_objectives("hardware", (_PRICED,), physics=_TRUENORTH)
    assert [o.name for o in objectives] == [_PRICED]
    assert objectives[0].goal == "min"
    with pytest.raises(ValueError, match=_PRICED):
        resolve_active_objectives("hardware", (_PRICED,), physics=None)


def test_the_legacy_axes_are_unaffected_by_the_gate():
    for physics in (None, _TRUENORTH):
        specs = resolve_active_specs("joint", ("total_params",), physics=physics)
        assert [spec.key for spec in specs] == ["total_params"]


def test_omitting_the_kwarg_keeps_the_capability_level_question():
    """A caller with no run context (the wizard's whole-catalog offer) must keep
    resolving exactly as before — None means 'declared none', not 'did not ask'."""
    specs = resolve_active_specs("joint", (_PRICED,))
    assert [spec.key for spec in specs] == [_PRICED]


def test_the_defaults_never_include_a_priced_axis():
    """Per-mode defaults stay byte-identical: the priced axes are opt-in by name,
    so a run that declares no physics still resolves its defaults."""
    for mode in ("model", "hardware", "joint"):
        specs = resolve_active_specs(mode, None, physics=None)
        assert not any(spec.key == _PRICED for spec in specs)
