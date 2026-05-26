"""Unit tests for the unified ``ComputeOpMapper`` and its ``ComputeAdapter`` payload."""

from __future__ import annotations

import operator

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.compute_modules import (
    ComputeAdapter,
    ScaleNormalizingWrapper,
    _cat_along,
)
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.mappers.perceptron import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import InputMapper


class TestUnary1DSource:
    def test_layer_norm_emits_single_compute_op(self):
        # InputMapper((16,)) normalises to (1, 16) — see InputMapper docstring.
        inp = InputMapper((16,))
        mapper = ComputeOpMapper(inp, nn.LayerNorm([16]))

        ir = IRMapping(
            q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
        )
        result = mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]

        assert len(compute_ops) == 1
        assert compute_ops[0].op_type == "LayerNorm"
        assert isinstance(compute_ops[0].params["module"], nn.LayerNorm)
        # Whole-tensor LayerNorm preserves the input shape.
        assert result.shape == (1, 16)

    def test_output_shape_auto_inferred(self):
        inp = InputMapper((20,))  # → (1, 20)
        mapper = ComputeOpMapper(inp, nn.Linear(20, 10))
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        # nn.Linear is per-instance, so per-column emission → output (1, 10).
        assert compute_ops[0].output_shape == (10,)


class TestUnary2DPerInstance:
    def test_linear_2d_per_column(self):
        # 4 tokens × 8 features each — Linear should be applied per token.
        inp = InputMapper((4, 8))
        # The Ensure2D layout convention is (instances=4, features=8).  We
        # construct the ComputeOpMapper directly so the source ndim is 2.
        mapper = ComputeOpMapper(inp, nn.Linear(8, 3))
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        result = mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        # 4 instances × 1 ComputeOp per instance.
        assert len(compute_ops) == 4
        assert all(op.op_type == "Linear" for op in compute_ops)
        # Output shape: (num_instances, out_features) = (4, 3).
        assert result.shape == (3, 4) or result.shape == (4, 3)


class TestUnary2DWholeTensor:
    def test_mean_emits_single_compute_op(self):
        inp = InputMapper((4, 8))  # 4 groups, 8 features
        mapper = ComputeOpMapper(
            inp, ComputeAdapter(torch.mean, kwargs={"dim": 1}),
        )
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        result = mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        assert len(compute_ops) == 1
        # Display label reports the wrapped callable, not "ComputeAdapter".
        assert "mean" in compute_ops[0].op_type.lower()
        assert compute_ops[0].input_shape == (4, 8)
        assert result.shape == (8,)

    def test_select_token(self):
        inp = InputMapper((5, 7))
        mapper = ComputeOpMapper(
            inp,
            ComputeAdapter(
                operator.getitem,
                extra_args=((slice(None), 0),),
            ),
        )
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        result = mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        assert len(compute_ops) == 1
        assert "getitem" in compute_ops[0].op_type
        assert result.shape == (7,)


class TestMultiInput:
    def test_two_input_add(self):
        a = InputMapper((4,))   # → (1, 4)
        b = InputMapper((4,))
        mapper = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        result = mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        assert len(compute_ops) == 1
        assert "add" in compute_ops[0].op_type
        # Sources are (1, 4); Add preserves shape → (1, 4).
        assert compute_ops[0].output_shape == (1, 4)


class TestBoundTensorPattern:
    """ComputeAdapter with bound tensors handles batch broadcasting on its own."""

    def test_constant_add_broadcasts(self):
        const = torch.tensor([10.0, 20.0, 30.0])  # batch-stripped
        adapter = ComputeAdapter(operator.add, bound_tensors=[const])

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (B=2, 3)
        out = adapter(x)
        expected = x + const  # torch auto-broadcasts; result must match
        assert torch.allclose(out, expected)

    def test_constant_prepend_via_cat_along(self):
        cls_token = torch.zeros(1, 4)  # one token, 4 features
        adapter = ComputeAdapter(
            _cat_along, bound_tensors=[cls_token], kwargs={"dim": 1},
        )
        x = torch.randn(3, 5, 4)  # (B=3, S=5, D=4)
        out = adapter(x)
        assert out.shape == (3, 6, 4)
        assert torch.allclose(out[:, 0], torch.zeros(3, 4))  # prepended zeros
        assert torch.allclose(out[:, 1:], x)


class TestScaleAwareEmission:
    """When ``per_source_scales`` is set, the underlying module is wrapped."""

    def test_unwrapped_when_scales_unset(self):
        a = InputMapper((3,))
        b = InputMapper((3,))
        mapper = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        mapper.map_to_ir(ir)
        op = [n for n in ir.nodes if isinstance(n, ComputeOp)][0]
        # No wrapper — params["module"] is the bare ComputeAdapter.
        assert isinstance(op.params["module"], ComputeAdapter)

    def test_wrapped_when_scales_set(self):
        a = InputMapper((3,))
        b = InputMapper((3,))
        mapper = ComputeOpMapper([a, b], ComputeAdapter(operator.add))
        mapper.per_source_scales = [torch.tensor([2.0, 2.0, 2.0]), torch.tensor([4.0, 4.0, 4.0])]
        mapper.output_scale = torch.tensor([3.0, 3.0, 3.0])
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        mapper.map_to_ir(ir)
        op = [n for n in ir.nodes if isinstance(n, ComputeOp)][0]
        # Wrapper present.
        assert isinstance(op.params["module"], ScaleNormalizingWrapper)
