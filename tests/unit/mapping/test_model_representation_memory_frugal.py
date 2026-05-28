"""ModelRepresentation.__call__ must release intermediate values once all consumers see them."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.model_representation import ModelRepresentation


class _PassThroughMapper(Mapper):
    def __init__(self, source):
        super().__init__(source_mapper=source)

    def _forward_impl(self, x):
        return x


class _RootMapper(Mapper):
    def __init__(self):
        super().__init__(source_mapper=None)

    def _forward_impl(self, x):
        return x


def _build_linear_chain(n: int):
    root = _RootMapper()
    head = root
    for _ in range(n):
        head = _PassThroughMapper(head)
    return ModelRepresentation(head)


class TestPeakLiveValues:
    def test_linear_chain_peak_is_bounded(self):
        n_layers = 32
        repr_ = _build_linear_chain(n_layers)
        x = torch.zeros(2, 4)
        out = repr_(x)
        assert out.shape == (2, 4)
        # For a linear chain, at most ~2 values should be live at any time:
        # the current node's input and its just-written output.
        assert hasattr(repr_, "_peak_live_values"), (
            "ModelRepresentation must expose _peak_live_values for memory introspection"
        )
        assert repr_._peak_live_values <= 3, (
            f"linear chain held {repr_._peak_live_values} intermediates; "
            "expected <= 3 with refcount-based cleanup"
        )

    def test_output_value_is_retained(self):
        repr_ = _build_linear_chain(4)
        x = torch.zeros(1, 3)
        out = repr_(x)
        # Output tensor must be returned correctly (cleanup must not delete the
        # final node's value before returning).
        assert out.shape == (1, 3)


class _FanOutMapper(Mapper):
    def __init__(self, source):
        super().__init__(source_mapper=source)

    def _forward_impl(self, x):
        return x + 1


class _AddTwoMapper(Mapper):
    def __init__(self, source_a, source_b):
        super().__init__(source_mapper=None)
        self._a = source_a
        self._b = source_b

    def get_source_mappers(self):
        return [self._a, self._b]

    def _forward_impl(self, xs):
        a, b = xs
        return a + b


class TestFanOutHandling:
    def test_fanout_node_value_lives_until_all_consumers_run(self):
        root = _RootMapper()
        branch_a = _FanOutMapper(root)
        branch_b = _FanOutMapper(root)
        merge = _AddTwoMapper(branch_a, branch_b)
        repr_ = ModelRepresentation(merge)
        x = torch.zeros(1, 4)
        out = repr_(x)
        assert torch.allclose(out, torch.full((1, 4), 2.0))
