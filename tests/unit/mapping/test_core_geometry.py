"""Tests for mapping.core_geometry."""

from types import SimpleNamespace

import pytest

from mimarsinan.mapping.core_geometry import used_axons, used_neurons


def _core(*, axons_per_core=8, available_axons=2, neurons_per_core=4, available_neurons=1):
    return SimpleNamespace(
        axons_per_core=axons_per_core,
        available_axons=available_axons,
        neurons_per_core=neurons_per_core,
        available_neurons=available_neurons,
    )


class TestCoreGeometry:
    def test_used_axons_raw(self):
        assert used_axons(_core()) == 6

    def test_used_neurons_raw(self):
        assert used_neurons(_core()) == 3

    def test_min_one_clamps_empty(self):
        c = _core(available_axons=8, available_neurons=4)
        assert used_axons(c, min_one=True) == 1
        assert used_neurons(c, min_one=True) == 1
