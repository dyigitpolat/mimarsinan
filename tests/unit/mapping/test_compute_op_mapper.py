"""Unit tests for the unified ``ComputeOpMapper`` and its compute-module wrappers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.compute_modules import (
    Add,
    ConstantAdd,
    Mean,
    ScaleNormalizingWrapper,
    Select,
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
        mapper = ComputeOpMapper(inp, Mean(dim=1))
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        result = mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        assert len(compute_ops) == 1
        assert compute_ops[0].op_type == "Mean"
        assert compute_ops[0].input_shape == (4, 8)
        assert result.shape == (8,)

    def test_select_token(self):
        inp = InputMapper((5, 7))
        mapper = ComputeOpMapper(inp, Select(index=0))
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        result = mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        assert len(compute_ops) == 1
        assert compute_ops[0].op_type == "Select"
        assert result.shape == (7,)


class TestMultiInput:
    def test_two_input_add(self):
        a = InputMapper((4,))   # → (1, 4)
        b = InputMapper((4,))
        mapper = ComputeOpMapper([a, b], Add())
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        result = mapper.map_to_ir(ir)
        compute_ops = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        assert len(compute_ops) == 1
        assert compute_ops[0].op_type == "Add"
        # Sources are (1, 4); Add preserves shape → (1, 4).
        assert compute_ops[0].output_shape == (1, 4)


class TestScaleAwareEmission:
    """When ``per_source_scales`` is set, the underlying module is wrapped."""

    def test_unwrapped_when_scales_unset(self):
        a = InputMapper((3,))
        b = InputMapper((3,))
        mapper = ComputeOpMapper([a, b], Add())
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        mapper.map_to_ir(ir)
        op = [n for n in ir.nodes if isinstance(n, ComputeOp)][0]
        # No wrapper — params["module"] is the bare Add.
        assert isinstance(op.params["module"], Add)

    def test_wrapped_when_scales_set(self):
        a = InputMapper((3,))
        b = InputMapper((3,))
        mapper = ComputeOpMapper([a, b], Add())
        mapper.per_source_scales = [torch.tensor([2.0, 2.0, 2.0]), torch.tensor([4.0, 4.0, 4.0])]
        mapper.output_scale = torch.tensor([3.0, 3.0, 3.0])
        ir = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
        mapper.map_to_ir(ir)
        op = [n for n in ir.nodes if isinstance(n, ComputeOp)][0]
        # Wrapper present.
        assert isinstance(op.params["module"], ScaleNormalizingWrapper)
