"""The MAC rule for functional weight ops (coverage W10): a host op evaluating a product in functional form.

A host op whose payload is a ``ComputeAdapter`` of ``F.linear``, ``matmul`` (``@``, ``torch.matmul``, the tensor
method), ``mm``, ``bmm`` or a two-operand ``einsum`` carries every output element times the extent the call sums over,
whether its weight is bound (a parameter, a buffer, a batch-free value) or an activation. On the base both counters
counted such an op 0, and the forward estimator's total left it out: Swin's qkv and output projections, inlined by W2
and W8 as ``F.linear`` on bound weights, made its host MAC share an understatement. The rule lives in the forward-MAC
SSOT (``module_macs``), so ``count_host_ops`` (the IR) and ``estimate_onchip_fraction(metric='macs')`` (the flow) read
it alike and agree; the total is checked against torch's own FLOP counter on the native model, an independent count.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformer, SwinTransformerBlockV2

from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.batch_free_value import BatchFreeValue
from mimarsinan.mapping.support.call_template import ArgTemplate, BoundSlot, LiteralSlot, SourceSlot
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.mapping.verification.onchip_majority import count_host_ops, module_macs
from mimarsinan.torch_mapping.converter import trace_and_convert


# ----------------------------------------------------------------------------- the rule
class TestRule:
    def test_linear_counts_every_output_element_times_in_features(self):
        adapter = ComputeAdapter(F.linear, bound_tensors=[torch.randn(20, 12), torch.randn(20)])
        assert module_macs(adapter, (1, 5, 12), (1, 5, 20)) == 5 * 20 * 12

    def test_linear_weight_given_by_keyword(self):
        template = ArgTemplate((SourceSlot(0),), (("weight", BoundSlot(0)), ("bias", BoundSlot(1))))
        adapter = ComputeAdapter(F.linear, bound_tensors=[torch.randn(7, 3), torch.randn(7)], arg_template=template)
        assert module_macs(adapter, (1, 4, 3), (1, 4, 7)) == 4 * 7 * 3

    def test_matmul_with_the_bound_weight_on_the_left(self):
        adapter = ComputeAdapter(torch.matmul, bound_tensors=[torch.randn(9, 5)],
                                 arg_template=ArgTemplate((BoundSlot(0), SourceSlot(0))))
        assert module_macs(adapter, (1, 5, 12), (1, 9, 12)) == 9 * 12 * 5

    @pytest.mark.parametrize("fn", [torch.matmul, torch.Tensor.matmul, __import__("operator").matmul],
                             ids=["torch", "method", "operator"])
    def test_matmul_with_the_bound_weight_on_the_right_in_every_spelling(self, fn):
        adapter = ComputeAdapter(fn, bound_tensors=[torch.randn(12, 7)])
        assert module_macs(adapter, (1, 5, 12), (1, 5, 7)) == 5 * 7 * 12

    def test_mm_and_bmm(self):
        assert module_macs(ComputeAdapter(torch.mm, bound_tensors=[torch.randn(60, 4)]), (1, 60), (1, 4)) == 240
        bmm = ComputeAdapter(torch.bmm, bound_tensors=[torch.randn(1, 6, 3)])
        assert module_macs(bmm, (1, 5, 6), (1, 5, 3)) == 5 * 3 * 6

    def test_a_two_operand_einsum_counts_its_summed_axes(self):
        adapter = ComputeAdapter(torch.einsum, bound_tensors=[torch.randn(12, 6)], extra_args=("bld,de->ble",),
                                 arg_template=ArgTemplate((LiteralSlot(0), SourceSlot(0), BoundSlot(0))))
        assert module_macs(adapter, (1, 5, 12), (1, 5, 6)) == 5 * 6 * 12

    def test_a_product_of_two_activations_in_functional_form_counts_too(self):
        adapter = ComputeAdapter(torch.matmul)
        assert module_macs(adapter, None, (1, 4, 6), [(1, 4, 3), (1, 3, 6)]) == 4 * 6 * 3

    def test_an_einsum_of_three_operands_is_no_single_product_and_counts_none(self):
        adapter = ComputeAdapter(torch.einsum, bound_tensors=[torch.randn(3, 3), torch.randn(3, 3)],
                                 extra_args=("bi,ij,jk->bk",),
                                 arg_template=ArgTemplate((LiteralSlot(0), SourceSlot(0), BoundSlot(0), BoundSlot(1))))
        assert module_macs(adapter, (1, 3), (1, 3)) == 0

    def test_a_call_that_is_no_product_counts_none(self):
        assert module_macs(ComputeAdapter(torch.relu), (1, 8), (1, 8)) == 0
        assert module_macs(ComputeAdapter(torch.add, bound_tensors=[torch.randn(8)]), (1, 8), (1, 8)) == 0

    def test_an_unknown_shape_counts_none(self):
        adapter = ComputeAdapter(F.linear, bound_tensors=[torch.randn(20, 12)])
        assert module_macs(adapter, None, (1, 5, 20)) == 0
        assert module_macs(adapter, (1, 5, 12), None) == 0

    def test_a_batch_free_slot_is_read_by_its_stored_shape_never_evaluated(self):
        class Refuses(nn.Module):
            def forward(self):
                raise AssertionError("the MAC rule evaluated a batch-free value")

        weight = BatchFreeValue(Refuses(), (20, 12))
        adapter = ComputeAdapter(F.linear, bound_tensors=[weight])
        assert module_macs(adapter, (1, 5, 12), (1, 5, 20)) == 5 * 20 * 12


# ----------------------------------------------------------------------------- both counters, the native count
def native_macs(model: nn.Module, shape) -> int:
    """The native model's MACs at batch 1 by torch's FLOP counter (FLOPs of mm, addmm, bmm, convolution / 2)."""
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, *shape))
    return int(counter.get_total_flops()) // 2


def counters(model: nn.Module, shape, classes: int):
    with contextlib.redirect_stdout(io.StringIO()):
        flow = trace_and_convert(model, shape, "cpu", packaging=MVM_PACKAGING).flow
        ir = IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
                       allow_coalescing=False, hardware_bias=False).map(flow.get_mapper_repr())
    return count_host_ops(ir), estimate_onchip_fraction(flow, shape, classes, metric="macs"), ir


class _Functional(nn.Module):
    """A Linear on chip, then one functional product with a bound weight on the host."""

    def __init__(self, form: str):
        super().__init__()
        self.form, self.inp = form, nn.Linear(8, 12)
        shapes = {"linear": (20, 12), "linear_kwargs": (20, 12), "at": (12, 7), "left": (9, 5), "method": (12, 7),
                  "einsum": (12, 6), "mm": (60, 4)}
        self.w = nn.Parameter(torch.randn(*shapes[form]))
        self.b = nn.Parameter(torch.randn(shapes[form][0]))

    def forward(self, x):
        y = torch.relu(self.inp(x))
        if self.form == "linear":
            return F.linear(y, self.w, self.b).sum(dim=1)
        if self.form == "linear_kwargs":
            return F.linear(y, weight=self.w, bias=self.b).sum(dim=1)
        if self.form == "at":
            return (y @ self.w).sum(dim=1)
        if self.form == "left":
            return torch.matmul(self.w, y).sum(dim=1)
        if self.form == "method":
            return y.matmul(self.w).sum(dim=1)
        if self.form == "einsum":
            return torch.einsum("bld,de->ble", y, self.w).sum(dim=1)
        return torch.mm(y.flatten(1), self.w)


@pytest.mark.parametrize("form", ["linear", "linear_kwargs", "at", "left", "method", "einsum", "mm"])
def test_both_counters_count_a_functional_weight_op_and_the_total_is_the_native_count(form):
    torch.manual_seed(0)
    model = _Functional(form).eval()
    host, est, _ = counters(model, (5, 8), 10)
    on_chip = 5 * 8 * 12
    assert host == est.host
    assert host == native_macs(model, (5, 8)) - on_chip > 0
    assert est.total == native_macs(model, (5, 8))


SWIN_SHAPE = (3, 16, 16)


def _swin(v2: bool) -> nn.Module:
    torch.manual_seed(0)
    extra = {"block": SwinTransformerBlockV2, "downsample_layer": PatchMergingV2} if v2 else {}
    return SwinTransformer(patch_size=[4, 4], embed_dim=8, depths=[2, 2], num_heads=[2, 2], window_size=[2, 2],
                           num_classes=5, **extra).eval()


def _projection_macs(model: nn.Module) -> int:
    """Every window attention's qkv and output projections at every token of its stage (the toy's windows tile the
    map, so no padding token): the products torchvision evaluates as ``F.linear`` on bound weights."""
    total, tokens = 0, (SWIN_SHAPE[1] // 4) * (SWIN_SHAPE[2] // 4)
    for stage in model.features[1::2]:
        for block in stage:
            for linear in (block.attn.qkv, block.attn.proj):
                total += tokens * linear.in_features * linear.out_features
        tokens //= 4
    return total


@pytest.mark.timeout(240)
def test_swins_projections_are_counted_and_the_total_is_the_native_count():
    model = _swin(v2=False)
    host, est, ir = counters(model, SWIN_SHAPE, 5)
    assert host == est.host
    linear_ops = [op for op in ir.get_compute_ops() if getattr(op.params["module"], "fn", None) is F.linear]
    assert linear_ops
    assert count_host_ops(IRGraph(nodes=linear_ops, output_sources=np.array([], dtype=object))) \
        == _projection_macs(model)
    assert est.total == native_macs(model, SWIN_SHAPE)


@pytest.mark.timeout(240)
def test_swin_v2_counts_its_projections_and_leaves_out_only_its_batch_free_products():
    """Swin V2's relative position bias (``cpb_mlp``, two Linears on a buffer) is a batch-free value: the deployed
    program holds it as a constant (``deployed_payload``), so neither counter counts its products, and the total is
    the native count less exactly those."""
    model = _swin(v2=True)
    host, est, _ = counters(model, SWIN_SHAPE, 5)
    assert host == est.host >= _projection_macs(model)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, *SWIN_SHAPE))
    per_module = counter.get_flop_counts()
    batch_free = sum(sum(per_module[name].values()) for name in per_module
                     if name.endswith("cpb_mlp") and name.count("cpb_mlp") == 1) // 2
    assert batch_free > 0
    assert est.total == native_macs(model, SWIN_SHAPE) - batch_free
