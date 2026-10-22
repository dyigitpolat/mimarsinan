"""Generic host ops keep the traced argument order (D11) and broadcast bound tensors by trailing alignment (D12).

Every witness converts a native fp64 model and compares the converted flow with
the native model at batch 1 and batch 3, relative error max|d| / max(1, max|ref|)
at most 1e-12; a subset also runs the MVM identity value program. Witnesses that
cannot keep the toolchain's batch-first contract are refused by name.
"""

from __future__ import annotations

import contextlib
import copy
import io
import operator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.mapper_graph_fx import (
    GenericOpArgumentError,
    GenericOpBatchAxisError,
)

# W differs from every tested batch so a forced (B, C) expansion cannot align with it by accident.
C, N, H, W = 6, 5, 4, 2
TOLERANCE = 1e-12
BATCHES = (1, 3)
PACKAGINGS = [pytest.param(MVM_PACKAGING, id="mvm"), pytest.param(SPIKING_PACKAGING, id="spiking")]


def _rand(*shape, positive=False):
    t = torch.randn(tuple(shape))
    return t.abs() + 0.5 if positive else t


class _Front(nn.Module):
    """An on-chip-mappable layer producing a rank-2 (B, C), rank-3 (B, n, C) or rank-4 (B, C, H, W) activation."""

    def __init__(self, rank):
        super().__init__()
        self.pre = nn.Conv2d(C, C, 1) if rank == 4 else nn.Linear(C, C)

    def forward(self, x):
        return self.pre(x)


def _input_shape(rank, tokens=N):
    return {2: (C,), 3: (tokens, C), 4: (C, H, W)}[rank]


def _rel_err(out, ref):
    return float((out - ref).abs().max()) / max(1.0, float(ref.abs().max()))


def _convert(model, input_shape, packaging):
    with contextlib.redirect_stdout(io.StringIO()):
        return convert_torch_model(
            copy.deepcopy(model), input_shape, num_classes=C, packaging=packaging,
        ).eval()


def _native(factory):
    torch.manual_seed(0)
    return factory().double().eval()


def _inputs(input_shape, batch):
    return torch.randn(batch, *input_shape, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(100 + batch))


def assert_converted_exact(factory, input_shape, packaging):
    model = _native(factory)
    flow = _convert(model, input_shape, packaging)
    for batch in BATCHES:
        x = _inputs(input_shape, batch)
        with torch.no_grad():
            ref, out = model(x), flow(x)
        assert out.shape == ref.shape, f"batch {batch}: {tuple(out.shape)} vs {tuple(ref.shape)}"
        err = _rel_err(out, ref)
        assert err <= TOLERANCE, f"batch {batch}: relative error {err:.3e}"


def assert_value_program_exact(factory, input_shape):
    model = _native(factory)
    flow = _convert(model, input_shape, MVM_PACKAGING)
    with contextlib.redirect_stdout(io.StringIO()):
        ir = map_model_to_ir(flow.get_mapper_repr())
    program = ValueHybridCoreFlow(
        build_identity_hybrid_mapping(ir_graph=ir), device="cpu", dtype=torch.float64,
    )
    for batch in BATCHES:
        x = _inputs(input_shape, batch)
        with torch.no_grad():
            ref, out = model(x), program(x)
        err = _rel_err(out.reshape(ref.shape), ref)
        assert err <= TOLERANCE, f"batch {batch}: value program relative error {err:.3e}"


# ── witness models ──────────────────────────────────────────────────────


class BinaryWithBound(nn.Module):
    """``op(c, y)`` or ``op(y, c)`` with a parameter ``c`` and the activation ``y``."""

    def __init__(self, rank, op, bound_shape, bound_left, positive=False):
        super().__init__()
        self.front = _Front(rank)
        self.op = op
        self.bound_left = bound_left
        self.c = nn.Parameter(_rand(*bound_shape, positive=positive))

    def forward(self, x):
        y = self.front(x)
        return self.op(self.c, y) if self.bound_left else self.op(y, self.c)


class SubAlpha(nn.Module):
    def __init__(self, rank, bound_left):
        super().__init__()
        self.front = _Front(rank)
        self.bound_left = bound_left
        self.c = nn.Parameter(_rand(W if rank == 4 else C))

    def forward(self, x):
        y = self.front(x)
        if self.bound_left:
            return torch.sub(self.c, y, alpha=2)
        return torch.sub(y, self.c, alpha=2)


class LiteralLeft(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.front = _Front(3)
        self.kind = kind

    def forward(self, x):
        y = self.front(x)
        if self.kind == "sub":
            return 2.0 - y
        if self.kind == "truediv":
            return 2.0 / y
        return 2.0 ** y


class ParamLeftMatmul(nn.Module):
    """``A @ y`` with a square parameter ``A`` over (B, n, n): the swapped product type-checks."""

    def __init__(self):
        super().__init__()
        self.front = _Front(3)
        self.a = nn.Parameter(_rand(C, C))

    def forward(self, x):
        return torch.matmul(self.a, self.front(x))


class LayerScaleTokens(nn.Module):
    """timm ``LayerScale``: ``x * gamma`` over (B, n, C)."""

    def __init__(self, gamma_left=False):
        super().__init__()
        self.front = _Front(3)
        self.gamma_left = gamma_left
        self.gamma = nn.Parameter(_rand(C))

    def forward(self, x):
        y = self.front(x)
        return self.gamma * y if self.gamma_left else y * self.gamma


class LayerScale2d(nn.Module):
    """torchvision ``CNBlock`` layer scale ``gamma * x`` over (B, C, H, W); ``channels_last`` is ConvNeXt v1's form."""

    def __init__(self, gamma_shape, channels_last=False):
        super().__init__()
        self.front = _Front(4)
        self.channels_last = channels_last
        self.gamma = nn.Parameter(_rand(*gamma_shape))

    def forward(self, x):
        y = self.front(x)
        if self.channels_last:
            return (self.gamma * y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.gamma * y


class ResMLPAffine(nn.Module):
    """ResMLP ``Affine``: ``alpha * x + beta``; ``addcmul`` is timm's ``torch.addcmul(beta, alpha, x)``."""

    def __init__(self, shape, addcmul=False):
        super().__init__()
        self.front = _Front(3)
        self.addcmul = addcmul
        self.alpha = nn.Parameter(_rand(*shape))
        self.beta = nn.Parameter(_rand(*shape))

    def forward(self, x):
        y = self.front(x)
        if self.addcmul:
            return torch.addcmul(self.beta, self.alpha, y)
        return self.alpha * y + self.beta


class KeywordBound(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.front = _Front(3)
        self.kind = kind
        self.w = nn.Parameter(_rand(C) + 2.0)
        self.b = nn.Parameter(_rand(C))
        self.m = nn.Parameter(_rand(C, C))

    def forward(self, x):
        y = self.front(x)
        if self.kind == "layer_norm":
            return F.layer_norm(y, (C,), weight=self.w, bias=self.b)
        if self.kind == "add_other":
            return torch.add(y, other=self.b)
        if self.kind == "clamp_min":
            return torch.clamp(y, min=self.b)
        return F.linear(y, weight=self.m, bias=self.b)


class FunctionalBound(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.front = _Front(3)
        self.kind = kind
        self.m = nn.Parameter(_rand(4, C))
        self.b = nn.Parameter(_rand(4))
        self.row = nn.Parameter(_rand(1, C))
        self.mt = nn.Parameter(_rand(C, 4))
        self.w = nn.Parameter(_rand(C) + 2.0)
        self.lb = nn.Parameter(_rand(C))

    def forward(self, x):
        y = self.front(x)
        if self.kind == "linear":
            return F.linear(y, self.m, self.b)
        if self.kind == "linear_single_output":
            return F.linear(y, self.row)
        if self.kind == "matmul_right":
            return torch.matmul(y, self.mt)
        return F.layer_norm(y, (C,), self.w, self.lb)


# ── D11: operand order ─────────────────────────────────────────────────


# (op, bound operand on the left, bound strictly positive); y ** c is not real-valued for a signed y.
_D11_CASES = {
    "c_sub_x": (operator.sub, True, False),
    "x_sub_c": (operator.sub, False, False),
    "c_div_x": (operator.truediv, True, False),
    "x_div_c": (operator.truediv, False, True),
    "c_pow_x": (operator.pow, True, True),
}


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("case", sorted(_D11_CASES))
def test_non_commutative_op_with_a_bound_operand(case, rank, packaging):
    op, bound_left, positive = _D11_CASES[case]
    bound_shape = (W,) if rank == 4 else (C,)
    assert_converted_exact(
        lambda: BinaryWithBound(rank, op, bound_shape, bound_left, positive=positive),
        _input_shape(rank), packaging,
    )


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("bound_left", [True, False], ids=["sub_c_x", "sub_x_c"])
def test_sub_with_alpha(bound_left, rank, packaging):
    assert_converted_exact(lambda: SubAlpha(rank, bound_left), _input_shape(rank), packaging)


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("kind", ["sub", "truediv", "pow"])
def test_literal_left_operand(kind, packaging):
    assert_converted_exact(lambda: LiteralLeft(kind), _input_shape(3), packaging)


@pytest.mark.parametrize("packaging", PACKAGINGS)
def test_parameter_left_matmul(packaging):
    assert_converted_exact(ParamLeftMatmul, _input_shape(3, tokens=C), packaging)


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("shape", [(C,), (1, 1, C)], ids=["dim", "published"])
def test_addcmul_affine(shape, packaging):
    assert_converted_exact(lambda: ResMLPAffine(shape, addcmul=True), _input_shape(3), packaging)


def test_value_program_keeps_the_operand_order():
    assert_value_program_exact(
        lambda: BinaryWithBound(3, operator.sub, (C,), bound_left=True), _input_shape(3),
    )
    assert_value_program_exact(lambda: LiteralLeft("truediv"), _input_shape(3))


# ── D12: bound tensors against activations of every rank ───────────────


def _bound_shape(rank, bound_rank):
    per_sample = _input_shape(rank)
    return per_sample[len(per_sample) - bound_rank:] if bound_rank else ()


_GRID_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "maximum": torch.maximum,
}
_RANK_GRID = [(rank, bound_rank) for rank in (2, 3, 4) for bound_rank in range(4) if bound_rank <= rank]


@pytest.mark.parametrize("bound_left", [True, False], ids=["c_op_x", "x_op_c"])
@pytest.mark.parametrize("op_name", sorted(_GRID_OPS))
@pytest.mark.parametrize("rank,bound_rank", _RANK_GRID, ids=[f"act{r}_bound{b}" for r, b in _RANK_GRID])
def test_elementwise_op_with_a_bound_tensor_of_any_rank(rank, bound_rank, op_name, bound_left):
    positive = op_name == "truediv"
    assert_converted_exact(
        lambda: BinaryWithBound(
            rank, _GRID_OPS[op_name], _bound_shape(rank, bound_rank), bound_left, positive=positive,
        ),
        _input_shape(rank), MVM_PACKAGING,
    )


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("gamma_left", [False, True], ids=["x_gamma", "gamma_x"])
@pytest.mark.parametrize("tokens", [N, BATCHES[-1]], ids=["tokens5", "tokens_equal_batch"])
def test_layerscale_over_tokens(tokens, gamma_left, packaging):
    assert_converted_exact(
        lambda: LayerScaleTokens(gamma_left), _input_shape(3, tokens=tokens), packaging,
    )


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize(
    "gamma_shape,channels_last",
    [((C, 1, 1), False), ((1, C, 1, 1), False), ((C,), True)],
    ids=["torchvision_c11", "lead_one_1c11", "channels_last_c"],
)
def test_layerscale_over_feature_maps(gamma_shape, channels_last, packaging):
    assert_converted_exact(
        lambda: LayerScale2d(gamma_shape, channels_last), _input_shape(4), packaging,
    )


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("shape", [(C,), (1, 1, C)], ids=["dim", "published"])
@pytest.mark.parametrize("tokens", [N, BATCHES[-1]], ids=["tokens5", "tokens_equal_batch"])
def test_resmlp_affine(tokens, shape, packaging):
    assert_converted_exact(lambda: ResMLPAffine(shape), _input_shape(3, tokens=tokens), packaging)


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("kind", ["layer_norm", "add_other", "clamp_min", "linear"])
def test_keyword_bound_tensor(kind, packaging):
    assert_converted_exact(lambda: KeywordBound(kind), _input_shape(3), packaging)


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("kind", ["linear", "linear_single_output", "matmul_right", "layer_norm"])
def test_functional_op_with_bound_tensors(kind, packaging):
    assert_converted_exact(lambda: FunctionalBound(kind), _input_shape(3), packaging)


def test_value_program_broadcasts_bound_tensors():
    assert_value_program_exact(lambda: LayerScaleTokens(gamma_left=True), _input_shape(3))
    assert_value_program_exact(lambda: ResMLPAffine((C,)), _input_shape(3, tokens=BATCHES[-1]))
    assert_value_program_exact(lambda: FunctionalBound("linear_single_output"), _input_shape(3))


# ── nested containers: sources and bound tensors inside list/tuple arguments ──


class StackOfSources(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = _Front(3)
        self.b = _Front(3)

    def forward(self, x):
        return torch.stack([self.a(x), self.b(x)], dim=1)


class IndexBufferGather(nn.Module):
    """``y[:, idx]`` with an index buffer: the bound tensor sits inside the index tuple."""

    def __init__(self):
        super().__init__()
        self.front = _Front(3)
        self.register_buffer("idx", torch.tensor([4, 0, 2]))

    def forward(self, x):
        return self.front(x)[:, self.idx]


@pytest.mark.parametrize("packaging", PACKAGINGS)
@pytest.mark.parametrize("factory", [StackOfSources, IndexBufferGather], ids=["stack", "index_buffer"])
def test_nested_container_arguments(factory, packaging):
    assert_converted_exact(factory, _input_shape(3), packaging)


def test_value_program_with_nested_container_arguments():
    assert_value_program_exact(StackOfSources, _input_shape(3))
    assert_value_program_exact(IndexBufferGather, _input_shape(3))


# ── refusals by name ───────────────────────────────────────────────────


class BoundOutranksActivation(nn.Module):
    """(B, C) + (1, 1, C) is (1, B, C) natively: the batch axis leaves position 0."""

    def __init__(self):
        super().__init__()
        self.front = _Front(2)
        self.c = nn.Parameter(_rand(1, 1, C))

    def forward(self, x):
        return self.front(x) + self.c


class DerivedConstantOperand(nn.Module):
    """``y * gamma[None, :]``: the operand is computed from a parameter, not read from one."""

    def __init__(self):
        super().__init__()
        self.front = _Front(2)
        self.gamma = nn.Parameter(_rand(C))

    def forward(self, x):
        return self.front(x) * self.gamma[None, :]


@pytest.mark.parametrize("packaging", PACKAGINGS)
def test_an_op_that_moves_the_batch_axis_is_refused_by_name(packaging):
    with pytest.raises(GenericOpBatchAxisError) as info:
        _convert(_native(BoundOutranksActivation), _input_shape(2), packaging)
    message = str(info.value)
    assert "generic op 'add' (FX node 'add')" in message
    assert "batch axis" in message
    assert info.value.node_name == "add"


@pytest.mark.parametrize("packaging", PACKAGINGS)
def test_an_unresolvable_operand_is_refused_by_name(packaging):
    with pytest.raises(GenericOpArgumentError) as info:
        _convert(_native(DerivedConstantOperand), _input_shape(2), packaging)
    message = str(info.value)
    assert "generic op 'mul' (FX node 'mul')" in message
    assert "argument 'getitem'" in message
    assert info.value.node_name == "mul"
    assert info.value.argument_name == "getitem"


class DeadShapeArithmetic(nn.Module):
    """``y.size(2) // w.size(0)`` only feeds a view's dimensions, so its op never runs."""

    def __init__(self):
        super().__init__()
        self.front = _Front(3)
        self.w = nn.Parameter(_rand(2, C))

    def forward(self, x):
        y = self.front(x)
        groups = y.size(2) // self.w.size(0)
        return y.view(y.size(0), y.size(1), groups, -1)


class LiveShapeQueryOperand(nn.Module):
    """``y * w.size(0)``: a parameter's shape query, static (its leading axis is not the batch axis)."""

    def __init__(self):
        super().__init__()
        self.front = _Front(2)
        self.w = nn.Parameter(_rand(3, C))

    def forward(self, x):
        return self.front(x) * self.w.size(0)


def _chain(exc):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__


@pytest.mark.parametrize("packaging", PACKAGINGS)
def test_unexecuted_shape_arithmetic_with_an_unresolvable_operand_still_converts(packaging):
    assert_converted_exact(DeadShapeArithmetic, _input_shape(3), packaging)


@pytest.mark.parametrize("packaging", PACKAGINGS)
def test_a_parameter_shape_query_operand_is_a_static_literal(packaging):
    """Restated in review round 2: ``w.size(0)`` is static, so the op converts exactly (it was refused before)."""
    assert_converted_exact(LiveShapeQueryOperand, _input_shape(2), packaging)
    flow = _convert(_native(LiveShapeQueryOperand), _input_shape(2), packaging)
    (mul,) = [m for m in flow.get_mapper_repr().execution_order() if getattr(m, "name", None) == "mul"]
    assert mul.module.extra_args == (3,)


def test_an_executed_op_with_an_unresolvable_non_tensor_operand_is_refused_by_name():
    """A non-tensor operand without a mapper (no conversion rule) refuses when the op runs, naming op and node."""
    import torch.fx as fx

    from mimarsinan.torch_mapping.mapper_graph_fx import _refusing_adapter

    graph = fx.Graph()
    x = graph.placeholder("x")
    unknown = graph.call_function(operator.add, (1, 2))
    mul = graph.call_function(operator.mul, (x, unknown))
    graph.output(mul)
    adapter = _refusing_adapter(operator.mul, mul, [unknown])
    with pytest.raises(RuntimeError) as info:
        adapter(torch.ones(2, 3))
    named = [e for e in _chain(info.value) if isinstance(e, GenericOpArgumentError)]
    assert named, f"no GenericOpArgumentError in the chain of {info.value!r}"
    assert named[0].node_name == "mul" and named[0].argument_name == "add"
    assert "generic op 'mul' (FX node 'mul')" in str(named[0])
