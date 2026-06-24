"""Tests for the on-chip parameter-majority validity gate.

The chip is only the deployment vehicle when the parameter MAJORITY is
physically placed on its crossbar cores. A mapping that offloads more than
half of the deployed model's parameters to host-side ComputeOps (the encoding
Linear/Conv, classifier readout, MultiheadAttention, etc.) is not a genuine
on-chip deployment and must be rejected at mapping time.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.verification.onchip_majority import (
    OnchipMajorityError,
    OnchipParamBreakdown,
    assert_onchip_majority_or_raise,
    compute_onchip_fraction,
    count_host_params,
)


def _input_sources(n):
    return np.array([IRSource(node_id=-2, index=i) for i in range(n)], dtype=object)


def _neural_core(core_id, axons, neurons, *, bias=False):
    """A NeuralCore carrying ``axons*neurons`` weights (+ ``neurons`` bias)."""
    w = np.random.randn(axons, neurons).astype(np.float32)
    hardware_bias = (
        np.random.randn(neurons).astype(np.float32) if bias else None
    )
    return NeuralCore(
        id=core_id,
        name=f"core_{core_id}",
        input_sources=_input_sources(axons),
        core_matrix=w,
        hardware_bias=hardware_bias,
    )


def _compute_op(op_id, module, n_inputs):
    return ComputeOp(
        id=op_id,
        name=f"op_{op_id}",
        input_sources=_input_sources(n_inputs),
        op_type=type(module).__name__,
        params={"module": module},
    )


def _graph(nodes):
    output_sources = np.array([IRSource(node_id=nodes[-1].id, index=0)], dtype=object)
    return IRGraph(nodes=list(nodes), output_sources=output_sources)


class TestCountHostParams:
    def test_param_free_compute_ops_count_zero(self):
        graph = _graph(
            [
                _neural_core(0, 8, 8),
                _compute_op(1, nn.ReLU(), 8),
                _compute_op(2, nn.MaxPool2d(2), 8),
            ]
        )
        assert count_host_params(graph) == 0

    def test_sums_module_parameters(self):
        # Linear(10, 4): 10*4 weight + 4 bias = 44 host params.
        graph = _graph([_compute_op(0, nn.Linear(10, 4), 10), _neural_core(1, 4, 4)])
        assert count_host_params(graph) == 44

    def test_dedups_shared_module_instance(self):
        # The same nn.Module wired into two ComputeOps must count ONCE.
        shared = nn.Linear(10, 4)  # 44 params
        graph = _graph(
            [
                _compute_op(0, shared, 10),
                _compute_op(1, shared, 10),
                _neural_core(2, 4, 4),
            ]
        )
        assert count_host_params(graph) == 44

    def test_counts_bound_tensors_when_no_module(self):
        bound = [torch.zeros(3, 5), torch.zeros(7)]
        op = ComputeOp(
            id=0,
            name="bound_op",
            input_sources=_input_sources(4),
            op_type="bound",
            params={"bound_tensors": bound},
        )
        graph = _graph([op, _neural_core(1, 4, 4)])
        assert count_host_params(graph) == 15 + 7


class TestComputeOnchipFraction:
    def test_decomposition_is_exact(self):
        # on-chip cores hold 8*8=64 weights; host Linear holds 44.
        graph = _graph([_neural_core(0, 8, 8), _compute_op(1, nn.Linear(10, 4), 10)])
        total = 64 + 44
        b = compute_onchip_fraction(graph, total_params=total)
        assert isinstance(b, OnchipParamBreakdown)
        assert b.host_params == 44
        assert b.onchip_params == 64
        assert b.total_params == total
        assert b.onchip_params + b.host_params == total
        assert b.fraction == pytest.approx(64 / total)

    def test_onchip_is_total_minus_host_not_physical_count(self):
        # Replicated cores (mlp_mixer tiling) inflate the PHYSICAL core weight
        # count far above total_params; the LOGICAL fraction stays <= 1 because
        # on-chip is defined as total - host, never the raw crossbar footprint.
        replicated = [_neural_core(i, 16, 16) for i in range(50)]  # 12800 phys w
        host = _compute_op(99, nn.Linear(4, 2), 4)  # 10 host params
        graph = _graph(replicated + [host])
        total = 200  # logical unique-param total (replication NOT in total)
        b = compute_onchip_fraction(graph, total_params=total)
        assert b.host_params == 10
        assert b.onchip_params == 190
        assert b.fraction == pytest.approx(190 / 200)
        assert 0.0 <= b.fraction <= 1.0


class TestAssertOnchipMajority:
    def test_onchip_majority_passes(self):
        # 64 on-chip vs 44 host => 59% on-chip => VALID.
        graph = _graph([_neural_core(0, 8, 8), _compute_op(1, nn.Linear(10, 4), 10)])
        b = assert_onchip_majority_or_raise(graph, total_params=64 + 44)
        assert b.fraction >= 0.5

    def test_host_majority_raises(self):
        # 16 on-chip vs 50240 host (a 784->64 encoding Linear) => 0.03% on-chip.
        graph = _graph(
            [_neural_core(0, 4, 4), _compute_op(1, nn.Linear(784, 64), 784)]
        )
        total = 16 + (784 * 64 + 64)
        with pytest.raises(OnchipMajorityError) as exc:
            assert_onchip_majority_or_raise(graph, total_params=total)
        msg = str(exc.value)
        assert "on-chip" in msg.lower()
        # The fraction must be named in the message.
        assert "%" in msg or "0." in msg

    def test_exactly_half_passes(self):
        # Boundary: exactly 50% on-chip is VALID (>= 0.5, not strict).
        # host = 9*10 + 10 bias = 100; on-chip = 10*10 = 100; total = 200.
        clean = _graph(
            [_neural_core(0, 10, 10), _compute_op(1, nn.Linear(9, 10), 9)]
        )
        out = assert_onchip_majority_or_raise(clean, total_params=200)
        assert out.fraction == pytest.approx(0.5)

    def test_custom_min_fraction(self):
        # 64 on-chip / 108 total = 0.59; a stricter 0.6 floor must reject it.
        graph = _graph([_neural_core(0, 8, 8), _compute_op(1, nn.Linear(10, 4), 10)])
        with pytest.raises(OnchipMajorityError):
            assert_onchip_majority_or_raise(graph, total_params=108, min_fraction=0.6)
