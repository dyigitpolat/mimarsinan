"""Unit tests for the on-chip residual-merge lowering pass (Tier 1).

Drives ``lower_residual_adds_to_onchip_merge`` directly on a converted mapper
graph: a param-free equal-width add becomes a merge ``PerceptronMapper`` (identity-
concat ``[I | I]`` weight, no trainable params), an unequal-width add is left as a
host ComputeOp, and a no-residual graph is untouched. Also locks the IRMapping
config gate (``onchip_residual_merge``): off → host add kept (byte-identical),
on → the add lowered onto an extra on-chip merge core.
"""

import operator

import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.support.residual_merge import (
    _ResidualConcatMapper,
    lower_residual_adds_to_onchip_merge,
)
from mimarsinan.torch_mapping.converter import convert_torch_model

INPUT_SHAPE = (12,)
NUM_CLASSES = 6
WIDTH = 10


class _EqualWidthResidual(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        self.stem = nn.Linear(12, width)
        self.a0 = nn.ReLU()
        self.f1 = nn.Linear(width, width)
        self.af1 = nn.ReLU()
        self.head = nn.Linear(width, NUM_CLASSES)

    def forward(self, x):
        z = self.a0(self.stem(x))
        b = self.af1(self.f1(z))
        return self.head(z + b)


class _NoResidual(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        self.stem = nn.Linear(12, width)
        self.a0 = nn.ReLU()
        self.head = nn.Linear(width, NUM_CLASSES)

    def forward(self, x):
        return self.head(self.a0(self.stem(x)))


def _repr_of(model):
    flow = convert_torch_model(model, INPUT_SHAPE, NUM_CLASSES, device="cpu")
    return flow.get_mapper_repr()


def _add_mappers(repr_):
    repr_._ensure_exec_graph()
    return [
        n for n in repr_._exec_order
        if isinstance(n, ComputeOpMapper)
        and isinstance(getattr(n, "module", None), ComputeAdapter)
        and n.module.fn is operator.add
    ]


def _merge_mappers(repr_):
    repr_._ensure_exec_graph()
    return [
        n for n in repr_._exec_order
        if isinstance(n, PerceptronMapper)
        and isinstance(n.source_mapper, _ResidualConcatMapper)
    ]


def test_equal_width_add_is_lowered_to_a_merge_perceptron():
    repr_ = _repr_of(_EqualWidthResidual())
    assert len(_add_mappers(repr_)) == 1
    n = lower_residual_adds_to_onchip_merge(repr_)
    assert n == 1
    assert len(_add_mappers(repr_)) == 0, "the host add must be gone after lowering"
    merges = _merge_mappers(repr_)
    assert len(merges) == 1


def test_merge_weight_is_frozen_identity_concat_no_bias():
    repr_ = _repr_of(_EqualWidthResidual())
    lower_residual_adds_to_onchip_merge(repr_)
    merge = _merge_mappers(repr_)[0].perceptron
    assert merge.layer.bias is None, "merge is param-free: no bias"
    w = merge.layer.weight.detach()
    assert tuple(w.shape) == (WIDTH, 2 * WIDTH)
    expected = torch.cat([torch.eye(WIDTH), torch.eye(WIDTH)], dim=1)
    assert torch.allclose(w, expected), "merge weight must be the identity-concat [I | I]"
    assert merge.layer.weight.requires_grad is False, "the merge bank is frozen"
    assert isinstance(merge.activation, nn.Identity), "merge is signed-IF (Identity, no ReLU)"


def test_merge_forward_sums_the_two_branches():
    repr_ = _repr_of(_EqualWidthResidual())
    lower_residual_adds_to_onchip_merge(repr_)
    merge = _merge_mappers(repr_)[0].perceptron
    z = torch.rand(3, WIDTH)
    b = torch.rand(3, WIDTH)
    concat = torch.cat([z, b], dim=1)
    assert torch.allclose(merge(concat), z + b, atol=1e-6)


def test_merge_adds_no_trainable_parameters():
    repr_ = _repr_of(_EqualWidthResidual())
    lower_residual_adds_to_onchip_merge(repr_)
    merge = _merge_mappers(repr_)[0].perceptron
    trainable = sum(p.numel() for p in merge.parameters() if p.requires_grad)
    assert trainable == 0, "the identity-merge core is param-free (no trainable params)"


def test_no_residual_graph_is_untouched():
    repr_ = _repr_of(_NoResidual())
    n = lower_residual_adds_to_onchip_merge(repr_)
    assert n == 0
    assert _merge_mappers(repr_) == []


def test_unequal_width_add_is_left_as_host_compute_op():
    """An unequal-width (projection) residual is NOT a bare add — the width guard
    leaves it a host ComputeOp. Force unequal widths on a real add to drive the
    guard branch directly."""
    repr_ = _repr_of(_EqualWidthResidual())
    adds = _add_mappers(repr_)
    assert len(adds) == 1
    add = adds[0]
    add.output_shape = None  # force width probe via sources
    from mimarsinan.mapping.support import residual_merge as rm
    orig = rm._branch_width
    widths = iter([WIDTH, WIDTH + 1])
    rm._branch_width = lambda m: next(widths)
    try:
        n = lower_residual_adds_to_onchip_merge(repr_)
    finally:
        rm._branch_width = orig
    assert n == 0, "an unequal-width (projection) residual is NOT a bare add — left host-side"
    assert len(_add_mappers(repr_)) == 1


# --- The IRMapping config gate: off keeps the host add; on lowers it ----------


def _ir_for(model, *, onchip_residual_merge):
    flow = convert_torch_model(model, INPUT_SHAPE, NUM_CLASSES, device="cpu")
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=1024, max_neurons=1024,
        onchip_residual_merge=onchip_residual_merge,
    )
    ir.map(repr_)
    return ir


def _add_compute_ops(ir):
    return [
        n for n in ir.nodes
        if isinstance(n, ComputeOp)
        and isinstance(n.params.get("module"), ComputeAdapter)
        and n.params["module"].fn is operator.add
    ]


def test_config_gate_off_keeps_host_add_compute_op():
    ir = _ir_for(_EqualWidthResidual(), onchip_residual_merge=False)
    assert len(_add_compute_ops(ir)) == 1, (
        "default (flag off) MUST keep the host ComputeOp add (byte-identical Tier-0)"
    )


def test_config_gate_on_lowers_add_to_one_extra_neural_core():
    ir_off = _ir_for(_EqualWidthResidual(), onchip_residual_merge=False)
    ir_on = _ir_for(_EqualWidthResidual(), onchip_residual_merge=True)
    assert len(_add_compute_ops(ir_on)) == 0, (
        "flag on MUST lower the residual add onto the crossbar — no host ComputeOp add"
    )
    n_off = len([n for n in ir_off.nodes if isinstance(n, NeuralCore)])
    n_on = len([n for n in ir_on.nodes if isinstance(n, NeuralCore)])
    assert n_on == n_off + 1, (
        f"flag on adds exactly one merge NeuralCore (got {n_off} -> {n_on})"
    )
