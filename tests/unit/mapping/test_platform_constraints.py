"""Tests for mapping.platform_constraints."""

from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params


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
