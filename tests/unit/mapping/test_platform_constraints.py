"""Tests for mapping.platform_constraints."""

import pytest

from mimarsinan.mapping.platform.platform_constraints import (
    bias_mode_for_cores,
    declares_hardware_bias,
    resolve_platform_mapping_params,
)


def test_legacy_bias_reserves_axon():
    cores = [{"max_axons": 256, "max_neurons": 128, "has_bias": False}]
    p = resolve_platform_mapping_params(cores)
    assert p.hardware_bias is False
    assert p.effective_max_axons == 255
    assert p.effective_max_neurons == 128


def test_hardware_bias_mode():
    cores = [{"max_axons": 256, "max_neurons": 128, "has_bias": True}]
    p = resolve_platform_mapping_params(cores)
    assert p.hardware_bias is True
    assert p.effective_max_axons == 256


class TestTheBiasQuestionIsTotal:
    """The config derivation asks it of a RAW draft grid, so no shape may raise
    and the ``max_axons``/``max_neurons`` precondition must not be inherited."""

    _UNJUDGEABLE = ["nope", b"nope", 7, 1.5, True, {"count": 2},
                    ["a", "b"], [{"has_bias": False}, "a"], (3, 4)]
    _NO_GRID = [None, [], (), {}, "", 0]

    @pytest.mark.parametrize("cores", _UNJUDGEABLE + _NO_GRID)
    def test_no_shape_raises(self, cores):
        assert isinstance(declares_hardware_bias(cores), bool)
        assert bias_mode_for_cores(cores) in ("on_chip", "param_encoded")

    @pytest.mark.parametrize("cores", _NO_GRID)
    def test_an_undeclared_grid_reads_as_the_framework_default_lane(self, cores):
        assert declares_hardware_bias(cores) is True
        assert bias_mode_for_cores(cores) == "on_chip"

    @pytest.mark.parametrize("cores", _UNJUDGEABLE)
    def test_an_unparseable_declaration_carries_no_lane(self, cores):
        assert declares_hardware_bias(cores) is False
        assert bias_mode_for_cores(cores) == "param_encoded"

    @pytest.mark.parametrize("cores,expected", [
        ([{}], True),
        ([{"count": 2}], True),
        ([{"has_bias": True}], True),
        ([{"has_bias": False}], False),
        ([{"has_bias": True}, {"has_bias": False}], False),
        ([{"max_axons": 256, "max_neurons": 128, "has_bias": True}], True),
        ([{"max_axons": 256, "max_neurons": 128, "has_bias": False}], False),
    ])
    def test_a_well_shaped_grid_reads_has_bias_with_its_default(self, cores, expected):
        assert declares_hardware_bias(cores) is expected
        assert bias_mode_for_cores(cores) == (
            "on_chip" if expected else "param_encoded"
        )

    @pytest.mark.parametrize("has_bias", [True, False])
    def test_the_dimensioned_grid_answers_exactly_as_the_mapping_params(self, has_bias):
        cores = [{"max_axons": 256, "max_neurons": 128, "has_bias": has_bias}]
        assert declares_hardware_bias(cores) is (
            resolve_platform_mapping_params(cores).hardware_bias
        )
