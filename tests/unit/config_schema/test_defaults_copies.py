"""Default getters must return copies: mutating a result never leaks into the module constants."""
from __future__ import annotations

import pytest

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_TRAINING_RECIPE,
    DEFAULT_TUNING_RECIPE,
    get_default_deployment_parameters,
    get_system_default_deployment_parameters,
    get_user_default_deployment_parameters,
)

_GETTERS = [
    get_default_deployment_parameters,
    get_user_default_deployment_parameters,
    get_system_default_deployment_parameters,
]


@pytest.mark.parametrize("getter", _GETTERS, ids=lambda g: g.__name__)
def test_top_level_result_is_a_copy(getter):
    out = getter()
    out["__sentinel__"] = object()
    assert "__sentinel__" not in DEFAULT_DEPLOYMENT_PARAMETERS
    assert "__sentinel__" not in getter()


@pytest.mark.parametrize("getter", _GETTERS, ids=lambda g: g.__name__)
@pytest.mark.parametrize("recipe_key", ["training_recipe", "tuning_recipe"])
def test_nested_recipe_blocks_are_copies(getter, recipe_key):
    out = getter()
    if recipe_key not in out:
        pytest.skip(f"{getter.__name__} does not expose {recipe_key}")
    recipe = out[recipe_key]
    assert isinstance(recipe, dict)
    recipe["__sentinel__"] = object()
    defaults_recipe = DEFAULT_DEPLOYMENT_PARAMETERS[recipe_key]
    assert isinstance(defaults_recipe, dict)
    assert "__sentinel__" not in defaults_recipe
    assert "__sentinel__" not in DEFAULT_TRAINING_RECIPE
    assert "__sentinel__" not in DEFAULT_TUNING_RECIPE
