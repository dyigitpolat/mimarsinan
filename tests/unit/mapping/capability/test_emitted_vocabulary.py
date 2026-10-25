"""The census classifies every host op the lowering emits: a tensor method by the torch function of its name.

G1's tensor-method rule converts ``x.softmax(dim=-1)`` as the host op of ``torch.Tensor.softmax`` and the batch-axis
push-forward rewrites ``qkv[i]`` into ``torch.Tensor.select``; both reach the census on G1's ViT-Lite (CCT-block) and
on every explicit-attention member (the coverage census rows ``vit_lite_7_4_*`` and ``net_models/vit_explicit_*``).
"""

from __future__ import annotations

import operator

import pytest
import torch
import torch.nn.functional as F

from mimarsinan.mapping.capability.ladder import LADDER
from mimarsinan.mapping.capability.member import AGREEMENT, convert_for_census, member_census
from mimarsinan.mapping.capability.op_kinds import ir_op_kinds, op_kind
from mimarsinan.mapping.capability.requirements import UnknownOpKindError
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.models.families.vit_lite import ViTLiteCCT
from mimarsinan.torch_mapping.converter_handlers.rules.method_rules import torch_function_of

from fixtures.requirement_census.scout_members import ExplicitViT

from .ir_fixtures import op, outputs

BITS = 8


def _unary(fn, width=4):
    return op(1, ComputeAdapter(fn), outputs(0, width), input_shape=(width,), output_shape=(width,))


def _binary(fn, width=4):
    return op(2, ComputeAdapter(fn), outputs(0, width) + outputs(1, width), input_shapes=((width,), (width,)),
              output_shape=(width,))


def _outcome(node):
    try:
        return op_kind(node)
    except UnknownOpKindError:
        return None


@pytest.mark.parametrize("method, kind", [
    (torch.Tensor.softmax, "softmax"), (torch.Tensor.select, "wire+"), (torch.Tensor.narrow, "wire+"),
    (torch.Tensor.relu, "clamp"), (torch.Tensor.sigmoid, "smooth"), (torch.Tensor.exp, "smooth"),
])
def test_a_tensor_method_takes_the_kind_of_the_torch_function_of_its_name(method, kind):
    assert op_kind(_unary(method)) == kind


def test_select_is_a_wire_op_in_both_spellings():
    assert op_kind(_unary(torch.select)) == "wire+"
    assert op_kind(_unary(torch.Tensor.select)) == "wire+"


def test_a_binary_tensor_method_reads_its_operand_structure_like_its_function():
    assert op_kind(_binary(torch.Tensor.mul)) == "prod"
    assert op_kind(op(1, ComputeAdapter(torch.Tensor.mul, bound_tensors=(torch.ones(4),)), outputs(0, 4),
                      input_shape=(4,), output_shape=(4,))) == "affine"


def test_every_tensor_method_with_a_torch_function_classifies_as_that_function():
    """One rule, no per-method rows: wherever the method rule has a torch function, both payloads agree (a kind or
    the same refusal), in the one-operand and in the two-operand structure."""
    methods = [name for name in dir(torch.Tensor) if torch_function_of(name) is not None]
    assert len(methods) > 100
    for name in methods:
        method, function = getattr(torch.Tensor, name), torch_function_of(name)
        assert _outcome(_unary(method)) == _outcome(_unary(function)), name
        assert _outcome(_binary(method)) == _outcome(_binary(function)), name


def test_a_tensor_method_with_no_torch_function_still_refuses_by_name():
    with pytest.raises(UnknownOpKindError, match="no op-kind rule"):
        op_kind(_unary(torch.Tensor.float))
    with pytest.raises(UnknownOpKindError, match="no op-kind rule"):
        op_kind(_unary(operator.not_))


def _vit_lite_cct(attention):
    return ViTLiteCCT(num_classes=5, img_size=8, patch=4, dim=16, depth=1, heads=2, attention=attention,
                      attn_drop=0.0, drop=0.0, stochastic_depth=0.0)


MEMBERS = {
    # G1's ViT-Lite (CCT-block) builder, the ViT gate's network and CL4's room-test member, in both attention forms.
    "vit_lite_cct_softmax": (lambda: _vit_lite_cct("softmax"), (3, 8, 8), 5),
    "vit_lite_cct_relu": (lambda: _vit_lite_cct("relu"), (3, 8, 8), 5),
    # The coverage census's timm-style member (net_models/vit_explicit_method_softmax): qkv[i] and x.softmax.
    "vit_explicit_method_softmax": (lambda: ExplicitViT(d=16, heads=2, depth=1, patch=8, softmax="method"),
                                    (3, 32, 32), 10),
}


def _member(key):
    factory, shape, classes = MEMBERS[key]
    torch.manual_seed(0)
    return factory().eval(), shape, classes


@pytest.mark.parametrize("key", sorted(MEMBERS))
def test_every_host_op_of_an_emitted_attention_member_has_a_kind(key):
    model, shape, classes = _member(key)
    converted = convert_for_census(model, shape, classes, packaging=MVM_PACKAGING)
    methods = {node.op_type for node in converted.ir.nodes
               if isinstance(node, ComputeOp) and "Tensor" in str(node.op_type)}
    assert methods, "the member's IR holds no tensor-method host op, so it does not exercise the method rule"
    kinds = ir_op_kinds(converted.ir)
    assert "wire+" in kinds.values()
    if key.endswith("softmax"):
        assert "softmax" in kinds.values()


@pytest.mark.timeout(240)
@pytest.mark.parametrize("key", sorted(MEMBERS))
def test_member_census_places_an_emitted_attention_member_on_every_level(key):
    model, shape, classes = _member(key)
    census = member_census(model, shape, classes, packaging=MVM_PACKAGING, value_bits=BITS)
    assert census.verdict.status == AGREEMENT, census.verdict
    assert census.census_refusal is None, census.census_refusal
    assert {row.level for row in census.rows} == {level.name for level in LADDER}


class _ConstantMaskAttention(torch.nn.Module):
    """Explicit attention with a constant (buffer) causal mask, in the method or the function form of masked_fill."""

    def __init__(self, form, d=8, n=4):
        super().__init__()
        self.form = form
        self.q, self.k, self.v = torch.nn.Linear(d, d), torch.nn.Linear(d, d), torch.nn.Linear(d, d)
        self.register_buffer("mask", torch.triu(torch.ones(n, n, dtype=torch.bool), 1))
        self.head = torch.nn.Linear(n * d, 3)

    def forward(self, x):
        s = self.q(x) @ self.k(x).transpose(-2, -1)
        s = s.masked_fill(self.mask, -1e9) if self.form == "method" else torch.masked_fill(s, self.mask, -1e9)
        return self.head(torch.flatten(s.softmax(-1) @ self.v(x), 1))


@pytest.mark.parametrize("form", ["method", "function"])
def test_a_constant_mask_attention_member_gets_rows_with_its_mask_on_the_host(form):
    """The merge's verify minor: masked_fill had no kind, so the census refused the member; it is the host-only mask
    kind now, named in every row, and the rest of the attention is placed as before."""
    torch.manual_seed(0)
    census = member_census(_ConstantMaskAttention(form).eval(), (4, 8), 3, packaging=MVM_PACKAGING, value_bits=BITS)
    assert census.verdict.status == AGREEMENT and census.census_refusal is None
    assert {row.level for row in census.rows} == {level.name for level in LADDER}
    assert all(row.host_kinds.get("mask") == 1 for row in census.rows), "a host op at every level"
