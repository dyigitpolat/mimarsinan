"""Tests for ``Mean``-via-``ComputeOpMapper`` IR mapping correctness.

Regression guard for the bug where the original ``MeanMapper._map_to_ir``
used [0] (selecting only the first group) instead of creating a ComputeOp
that averages all groups.  After the ComputeOpMapper unification the
``Mean(nn.Module)`` is wrapped by ``ComputeOpMapper`` and the executor
reshapes the gathered ``(B, N)`` flat input back to the structured
``input_shape`` before reducing.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.compute_modules import Mean
from mimarsinan.mapping.mappers.perceptron import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import ComputeOp, IRSource


class TestMeanMapperIRMapping:
    """Verify the Mean ComputeOpMapper emits a single reducing ComputeOp."""

    def test_creates_compute_op(self):
        """_map_to_ir must return sources from a ComputeOp, not raw subscript."""
        inp = InputMapper((4, 8))  # 4 groups, 8 features each
        mean = ComputeOpMapper(inp, Mean(dim=1))

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        result = mean.map_to_ir(ir_mapping)

        # Result should be 1D array of 8 IRSource objects
        assert result.shape == (8,), f"Expected shape (8,), got {result.shape}"

        # Each source should point to a ComputeOp node
        # (node_id should be a valid non-input node)
        for src in result.flatten():
            assert isinstance(src, IRSource)
            assert src.node_id >= 0, (
                f"IRSource(node_id={src.node_id}) looks like an input source, "
                "not a ComputeOp output."
            )

    def test_compute_op_has_all_groups_as_inputs(self):
        """The mean ComputeOp must wire ALL groups, not just group 0."""
        inp = InputMapper((4, 8))  # 4 groups, 8 features
        mean = ComputeOpMapper(inp, Mean(dim=1))

        ir_mapping = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        result = mean.map_to_ir(ir_mapping)

        # Output shape: 8 features after mean over 4 groups.
        assert result.shape == (8,)

        # All output sources should come from the same ComputeOp node.
        node_ids = set(src.node_id for src in result.flatten())
        assert len(node_ids) == 1, f"Expected 1 ComputeOp node, got {len(node_ids)} distinct nodes"

    def test_compute_op_executes_mean_correctly(self):
        """The Mean ComputeOp must actually compute the mean, not select [0]."""
        op = ComputeOp(
            id=0,
            name="test_mean",
            input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(12)]),
            op_type="module",
            params={"module": Mean(dim=1), "input_shape": (3, 4)},
            input_shape=(3, 4),
            output_shape=(4,),
        )

        # 3 groups of 4 features → mean over groups
        # Group 0: [1, 2, 3, 4], Group 1: [5, 6, 7, 8], Group 2: [9, 10, 11, 12]
        flat_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.float32)
        result = op.execute_on_gathered(flat_input)

        expected = torch.tensor([[5.0, 6.0, 7.0, 8.0]])  # mean of each feature across groups
        assert torch.allclose(result, expected, atol=1e-6), (
            f"Mean ComputeOp result {result} != expected {expected}"
        )


class TestMeanMapperForward:
    """Verify the ComputeOpMapper(Mean) _forward_impl computes the true mean."""

    def test_forward_computes_mean(self):
        inp = InputMapper((4, 8))
        mean = ComputeOpMapper(inp, Mean(dim=1))

        x = torch.randn(2, 4, 8)
        result = mean.forward(x)
        expected = x.mean(dim=1)
        assert torch.allclose(result, expected)
        assert result.shape == (2, 8)
