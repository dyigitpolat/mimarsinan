"""Regression: ``ComputeOpMapper(Mean)`` must average over all groups, not select [0]."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.ir import ComputeOp, IRSource


def _Mean(dim):
    return ComputeAdapter(torch.mean, kwargs={"dim": dim})


class TestMeanMapperIRMapping:
    def test_creates_compute_op(self):
        inp = InputMapper((4, 8))
        mean = ComputeOpMapper(inp, _Mean(dim=1))

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        result = mean.map_to_ir(ir_mapping)

        assert result.shape == (8,)

        for src in result.flatten():
            assert isinstance(src, IRSource)
            assert src.node_id >= 0, (
                f"IRSource(node_id={src.node_id}) looks like an input source, "
                "not a ComputeOp output."
            )

    def test_compute_op_has_all_groups_as_inputs(self):
        inp = InputMapper((4, 8))
        mean = ComputeOpMapper(inp, _Mean(dim=1))

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        result = mean.map_to_ir(ir_mapping)

        assert result.shape == (8,)

        node_ids = set(src.node_id for src in result.flatten())
        assert len(node_ids) == 1

    def test_compute_op_executes_mean_correctly(self):
        op = ComputeOp(
            id=0,
            name="test_mean",
            input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(12)]),
            op_type="module",
            params={"module": _Mean(dim=1), "input_shape": (3, 4)},
            input_shape=(3, 4),
            output_shape=(4,),
        )

        flat_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.float32)
        result = op.execute_on_gathered(flat_input)

        expected = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        assert torch.allclose(result, expected, atol=1e-6)


class TestMeanMapperForward:
    def test_forward_computes_mean(self):
        inp = InputMapper((4, 8))
        mean = ComputeOpMapper(inp, _Mean(dim=1))

        x = torch.randn(2, 4, 8)
        result = mean.forward(x)
        expected = x.mean(dim=1)
        assert torch.allclose(result, expected)
        assert result.shape == (2, 8)
