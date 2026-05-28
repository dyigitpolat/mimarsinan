"""ComputeOpMapper must reject broadcast-incompatible multi-input shapes before allocating."""

from __future__ import annotations

import operator

import pytest
import torch

from mimarsinan.mapping.mappers.compute_op_mapper import (
    ComputeOpMapper,
    ShapeMismatchError,
)
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter


class TestRuntimeBroadcastGuard:
    def test_incompatible_shapes_raise_shape_mismatch(self):
        mapper = ComputeOpMapper(
            [InputMapper((3,)), InputMapper((4,))],
            ComputeAdapter(operator.add),
            name="bad_add",
        )
        a = torch.zeros(2, 3)
        b = torch.zeros(2, 4)
        with pytest.raises(ShapeMismatchError) as excinfo:
            mapper.forward((a, b))
        msg = str(excinfo.value)
        assert "bad_add" in msg
        # Both observed shapes must appear in the error so debugging is possible.
        assert "(2, 3)" in msg or "[2, 3]" in msg or "2, 3" in msg
        assert "(2, 4)" in msg or "[2, 4]" in msg or "2, 4" in msg

    def test_compatible_shapes_pass_through(self):
        mapper = ComputeOpMapper(
            [InputMapper((4,)), InputMapper((4,))],
            ComputeAdapter(operator.add),
            name="ok_add",
        )
        a = torch.ones(2, 4)
        b = torch.ones(2, 4)
        out = mapper.forward((a, b))
        assert torch.allclose(out, torch.full((2, 4), 2.0))

    def test_broadcast_compatible_shapes_pass(self):
        mapper = ComputeOpMapper(
            [InputMapper((4,)), InputMapper((1,))],
            ComputeAdapter(operator.add),
            name="bcast_add",
        )
        a = torch.zeros(2, 4)
        b = torch.zeros(2, 1)
        out = mapper.forward((a, b))
        assert out.shape == (2, 4)

    def test_unary_mapper_is_unaffected(self):
        """Single-source mappers must not pay any cost for the guard."""
        import torch.nn as nn

        mapper = ComputeOpMapper(
            InputMapper((4,)), nn.Linear(4, 2), name="lin",
        )
        a = torch.zeros(3, 4)
        out = mapper.forward(a)
        assert out.shape == (3, 2)
