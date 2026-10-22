"""Value-parity conversion probe (E270): recorded exactness witnesses, mutations and exact programs.

Every witness pins its outcome on the integrated tree: exact within 1e-12, or refused
by the named refusal type it must raise. The witnesses are the recorded ones of the
significance review (D1, D2, D8 from ``audit_v42/arith/arith_models.py``; D9 to D12
from ``08_significance/probes/net``), ported verbatim and built inside the test.
"""

from __future__ import annotations

import copy
from typing import Callable

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import (
    MVM_PACKAGING,
    SPIKING_PACKAGING,
    PackagingContract,
)
from mimarsinan.models.families.densenet_bc import CifarDenseNetBC
from mimarsinan.models.families.pyramidnet import CifarPyramidNet
from mimarsinan.torch_mapping import converter
from mimarsinan.torch_mapping.conversion_probe import probe_forward
from mimarsinan.torch_mapping.converter import (
    TracedConversion,
    convert_torch_model,
    probe_witness_conversion,
    trace_and_convert,
)
from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityError
from mimarsinan.torch_mapping.value_parity import (
    ConversionValueParityError,
    probe_inputs,
    randomize_batchnorm_statistics,
    relative_error,
)
from mimarsinan.torch_mapping.value_parity.localization import mapper_label
from mimarsinan.torch_mapping.value_parity.probe import is_toolchain_refusal
from mimarsinan.torch_mapping.value_parity.value_program import identity_value_program

EXACT = 1e-12
PACKAGINGS = {"mvm": MVM_PACKAGING, "spiking": SPIKING_PACKAGING}
VSHAPE = (3, 32, 32)
SSHAPE = (16, 32)


# ----------------------------------------------------------------------------- witnesses (ported)
class BnAfterAct(nn.Module):
    """D1: conv -> ReLU -> BN -> conv -> ReLU -> FC (BN after the activation)."""

    def __init__(self):
        super().__init__()
        self.c1, self.a1, self.bn = nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8)
        self.c2, self.a2 = nn.Conv2d(8, 8, 3, padding=1, stride=2), nn.ReLU()
        self.fc = nn.Linear(8 * 16 * 16, 100)

    def forward(self, x):
        x = self.a2(self.c2(self.bn(self.a1(self.c1(x)))))
        return self.fc(torch.flatten(x, 1))


class PaddingModeWitness(nn.Module):
    """D2: Conv2d with a non-zero padding mode."""

    def __init__(self, mode="circular"):
        super().__init__()
        self.c1, self.a1 = nn.Conv2d(3, 8, 3, padding=1, padding_mode=mode), nn.ReLU()
        self.fc = nn.Linear(8 * 32 * 32, 100)

    def forward(self, x):
        return self.fc(torch.flatten(self.a1(self.c1(x)), 1))


class FanoutBN(nn.Module):
    """D8: the conv output feeds BN-ReLU-conv and, raw, the skip of an add."""

    def __init__(self):
        super().__init__()
        self.c0, self.bn, self.r = nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU()
        self.c1, self.r1 = nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        self.fc = nn.Linear(8 * 32 * 32, 100)

    def forward(self, x):
        h = self.c0(x)
        return self.fc(torch.flatten(self.r1(self.c1(self.r(self.bn(h))) + h), 1))


class FanoutAct(nn.Module):
    """D8 (spiking): the conv output feeds ReLU-conv and, raw, the skip of an add."""

    def __init__(self):
        super().__init__()
        self.c0, self.r = nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()
        self.c1, self.r1 = nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        self.fc = nn.Linear(8 * 32 * 32, 100)

    def forward(self, x):
        h = self.c0(x)
        return self.fc(torch.flatten(self.r1(self.c1(self.r(h)) + h), 1))


class Conv1dWitness(nn.Module):
    """D9: Conv1d(16 -> 8) on a (C=16, L=32) input, flatten, FC."""

    def __init__(self, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv1d(16, 8, k, padding=p)
        self.fc = nn.Linear(8 * (32 + 2 * p - k + 1), 10)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))


class BnEpsWitness(nn.Module):
    """D10: conv (or Linear) -> BN(eps) -> ReLU -> FC."""

    def __init__(self, eps, lin=False):
        super().__init__()
        self.lin = lin
        self.c = nn.Linear(48, 32) if lin else nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm1d(32, eps=eps) if lin else nn.BatchNorm2d(8, eps=eps)
        self.r = nn.ReLU()
        self.fc = nn.Linear(32 if lin else 8 * 16 * 16, 10)

    def forward(self, x):
        if self.lin:
            x = torch.flatten(x, 1)[:, :48]
        return self.fc(torch.flatten(self.r(self.bn(self.c(x))), 1))


class OpWitness(nn.Module):
    """conv(3->8) -> OP -> conv(8->8) -> ReLU -> flatten -> FC; the OP is the probed operation."""

    def __init__(self, op: nn.Module, conv_kw=None, spatial_after=32, ch_after=8):
        super().__init__()
        kw = {"padding": 1}
        kw.update(conv_kw or {})
        self.c1 = nn.Conv2d(3, 8, 3, **kw)
        self.op = op
        self.c2 = nn.Conv2d(ch_after, 8, 3, padding=1)
        self.a2 = nn.ReLU()
        self.fc = nn.Linear(8 * spatial_after * spatial_after, 10)

    def forward(self, x):
        return self.fc(torch.flatten(self.a2(self.c2(self.op(self.c1(x)))), 1))


class ParamFirstSub(nn.Module):
    """D11: c - x with a full-shape parameter c on the LEFT."""

    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.randn(1, 8, 32, 32))

    def forward(self, x):
        return self.c - x


class LayerScaleTokens(nn.Module):
    """D12: Linear on (B, N, C) tokens, a per-channel parameter multiply (LayerScale), mean, head."""

    def __init__(self):
        super().__init__()
        self.l = nn.Linear(32, 32)
        self.g = nn.Parameter(torch.rand(32))
        self.h = nn.Linear(32, 10)

    def forward(self, x):
        return self.h((self.g * self.l(x)).mean(dim=1))


class AbsorbedActivation(nn.Module):
    """Linear -> ACT -> Linear: an absorbable activation with a non-default parameter."""

    def __init__(self, act: nn.Module):
        super().__init__()
        self.l = nn.Linear(16, 16)
        self.a = act
        self.h = nn.Linear(16, 4)

    def forward(self, x):
        return self.h(self.a(self.l(x)))


Factory = Callable[[], nn.Module]

EXACT_OUTCOME = "exact"

WITNESSES: list[tuple[str, Factory, tuple[int, ...], object]] = [
    ("D1_bn_after_act", BnAfterAct, VSHAPE, EXACT_OUTCOME),
    ("D2_circular", lambda: PaddingModeWitness("circular"), VSHAPE, RepresentabilityError),
    ("D2_reflect", lambda: PaddingModeWitness("reflect"), VSHAPE, RepresentabilityError),
    ("D2_replicate", lambda: PaddingModeWitness("replicate"), VSHAPE, RepresentabilityError),
    ("D8_fanout_bn", FanoutBN, VSHAPE, EXACT_OUTCOME),
    ("D8_fanout_act", FanoutAct, VSHAPE, EXACT_OUTCOME),
    ("D9_conv1d_k3p1", lambda: Conv1dWitness(3, 1), (16, 32), EXACT_OUTCOME),
    ("D9_conv1d_k1p0", lambda: Conv1dWitness(1, 0), (16, 32), EXACT_OUTCOME),
    ("D10_conv_bn_eps_1e-3", lambda: BnEpsWitness(1e-3), (3, 16, 16), EXACT_OUTCOME),
    ("D10_conv_bn_eps_1e-1", lambda: BnEpsWitness(1e-1), (3, 16, 16), EXACT_OUTCOME),
    ("D11_param_first_sub", lambda: OpWitness(ParamFirstSub()), VSHAPE, EXACT_OUTCOME),
    ("D12_layerscale_tokens", LayerScaleTokens, SSHAPE, EXACT_OUTCOME),
    ("absorbed_leaky_relu_slope_0.2", lambda: AbsorbedActivation(nn.LeakyReLU(0.2)), (16,), EXACT_OUTCOME),
    ("absorbed_gelu_tanh", lambda: AbsorbedActivation(nn.GELU(approximate="tanh")), (16,), EXACT_OUTCOME),
    # The E270 family instances (D1 and D8 at once), built from the families at the code head.
    ("family_pyramidnet_d8_a8", lambda: CifarPyramidNet(VSHAPE, 10, depth=8, alpha=8), VSHAPE, EXACT_OUTCOME),
    ("family_densenet_bc_l10_k4_published",
     lambda: CifarDenseNetBC(VSHAPE, 10, depth=10, growth_rate=4, layer_order="published"), VSHAPE,
     EXACT_OUTCOME),
]


# ----------------------------------------------------------------------------- the corpus oracle
def _witness(factory: Factory) -> nn.Module:
    torch.manual_seed(0)
    model = factory().double().eval()
    randomize_batchnorm_statistics(model, seed=1)
    return model


def _refusal_or_raise(exc: Exception):
    if not is_toolchain_refusal(exc):
        raise exc
    return "refused", exc


def _converted_outcome(model: nn.Module, shape, packaging: PackagingContract):
    """("refused", exc) or ("measured", relative error of the returned flow, measured independently)."""
    reference = copy.deepcopy(model)
    x = probe_inputs(shape, seed=7, batch=4)
    try:
        flow = convert_torch_model(model, shape, 10, packaging=packaging)
    except Exception as exc:
        return _refusal_or_raise(exc)
    with torch.no_grad():
        return "measured", relative_error(flow.eval()(x), reference(x)).relative_error


def _value_program_outcome(model: nn.Module, shape):
    """("refused", exc) or ("measured", relative error of the identity value program, measured independently)."""
    reference = copy.deepcopy(model)
    x = probe_inputs(shape, seed=7, batch=4)
    try:
        probe_witness_conversion(copy.deepcopy(model), shape, packaging=MVM_PACKAGING)
    except Exception as exc:
        return _refusal_or_raise(exc)
    flow = trace_and_convert(model, shape, "cpu", packaging=MVM_PACKAGING).flow
    with torch.no_grad():
        want = reference(x)
        return "measured", relative_error(identity_value_program(flow)(x).reshape(want.shape), want).relative_error


def _assert_pinned(outcome, expected) -> None:
    kind, value = outcome
    if expected == EXACT_OUTCOME:
        assert kind == "measured", f"an exact witness was refused: {value!r}"
        assert value <= EXACT, f"accepted with relative error {value:.3e} > {EXACT:g}"
        return
    assert kind == "refused" and isinstance(value, expected), f"expected a {expected.__name__}, got {value!r}"


@pytest.mark.parametrize("packaging_name", sorted(PACKAGINGS))
@pytest.mark.parametrize("name,factory,shape,expected", WITNESSES, ids=[w[0] for w in WITNESSES])
def test_witness_converted_model_outcome_is_pinned(name, factory, shape, expected, packaging_name):
    _assert_pinned(_converted_outcome(_witness(factory), shape, PACKAGINGS[packaging_name]), expected)


@pytest.mark.parametrize("name,factory,shape,expected", WITNESSES, ids=[w[0] for w in WITNESSES])
def test_witness_identity_value_program_outcome_is_pinned(name, factory, shape, expected):
    _assert_pinned(_value_program_outcome(_witness(factory), shape), expected)


# ----------------------------------------------------------------------------- mutations
class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.r1 = nn.ReLU()
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        return self.fc(self.r1(self.l1(x)))


class _TwoBranchSub(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(12, 8)
        self.b = nn.Linear(12, 8)
        self.h = nn.Linear(8, 4)

    def forward(self, x):
        return self.h(torch.relu(self.a(x) - self.b(x)))


class _ConvBNReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 6, 3, padding=1)
        self.bn = nn.BatchNorm2d(6)
        self.r = nn.ReLU()
        self.fc = nn.Linear(6 * 8 * 8, 5)

    def forward(self, x):
        return self.fc(torch.flatten(self.r(self.bn(self.c1(x))), 1))


class _LinBNReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.bn = nn.BatchNorm1d(16)
        self.r = nn.ReLU()
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        return self.fc(self.r(self.bn(self.l1(x))))


class _BiasAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.shift = nn.Parameter(torch.randn(1, 16))
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        return self.fc(torch.relu(self.l1(x)) + self.shift)


def _mapper(flow, label: str):
    return next(n for n in flow.get_mapper_repr().execution_order() if mapper_label(n) == label)


def _linear_layer(mapper) -> nn.Module:
    perceptron = getattr(mapper, "perceptron", None)
    if perceptron is not None:
        return perceptron.layer
    module = mapper.module
    return module[0] if isinstance(module, nn.Sequential) else module


class _SwappedOperands(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *inputs, **kwargs):
        return self.inner(*reversed(inputs), **kwargs)


class _FirstSampleBroadcast(nn.Module):
    """A broadcast that is right at batch 1 only: every sample receives sample 0's value."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *inputs, **kwargs):
        out = self.inner(*inputs, **kwargs)
        return out[:1].expand_as(out)


def _negate_fc(flow):
    with torch.no_grad():
        _linear_layer(_mapper(flow, "fc")).weight.neg_()


def _drop_fc_bias(flow):
    with torch.no_grad():
        _linear_layer(_mapper(flow, "fc")).bias.zero_()


def _swap_sub_operands(flow):
    mapper = _mapper(flow, "sub")
    mapper.module = _SwappedOperands(mapper.module)


def _drop_eps(flow):
    for module in flow.modules():
        if isinstance(getattr(module, "eps", None), float):
            module.eps = torch.finfo(torch.float64).tiny  # F.batch_norm refuses eps == 0


def _transpose_c1_kernel(flow):
    mapper = _mapper(flow, "c1")
    weight, (kh, kw) = mapper.perceptron.layer.weight, mapper.kernel_size
    with torch.no_grad():
        kernel = weight.data.view(weight.shape[0], -1, kh, kw).transpose(2, 3)
        weight.copy_(kernel.reshape(weight.shape[0], -1))


def _broadcast_add_from_first_sample(flow):
    mapper = _mapper(flow, "add")
    mapper.module = _FirstSampleBroadcast(mapper.module)


MUTATIONS = [
    # id, packaging, model factory, input shape, mutation of the converted flow, expected first diverging node
    ("wrong_sign", "mvm", _MLP, (12,), _negate_fc, "fc"),
    ("wrong_sign_spiking", "spiking", _MLP, (12,), _negate_fc, "fc"),
    ("dropped_bias", "mvm", _MLP, (12,), _drop_fc_bias, "fc"),
    ("swapped_operand", "mvm", _TwoBranchSub, (12,), _swap_sub_operands, "sub"),
    ("dropped_eps_conv", "mvm", _ConvBNReLU, (3, 8, 8), _drop_eps, "c1"),
    ("dropped_eps_linear", "spiking", _LinBNReLU, (12,), _drop_eps, "l1"),
    ("wrong_transpose", "spiking", _ConvBNReLU, (3, 8, 8), _transpose_c1_kernel, "c1"),
    ("batch_dependent_broadcast", "mvm", _BiasAdd, (12,), _broadcast_add_from_first_sample, "add"),
]


def _make_conversion_wrong(monkeypatch, mutate) -> None:
    """Every flow the conversion returns is mutated, so the probe's witness conversion is wrong too."""
    original = converter.trace_and_convert

    def wrong_conversion(*args, **kwargs) -> TracedConversion:
        traced = original(*args, **kwargs)
        mutate(traced.flow)
        return traced

    monkeypatch.setattr(converter, "trace_and_convert", wrong_conversion)


@pytest.mark.parametrize(
    "name,packaging_name,factory,shape,mutate,node", MUTATIONS, ids=[m[0] for m in MUTATIONS],
)
def test_probe_refuses_deliberately_wrong_conversion(
    monkeypatch, name, packaging_name, factory, shape, mutate, node,
):
    """The conversion itself is made wrong (every flow it returns is mutated), so the probe's witness is too."""
    packaging = PACKAGINGS[packaging_name]
    exact = convert_torch_model(_witness(factory), shape, 6, packaging=packaging)
    assert exact is not None, "the unmutated program converts"

    _make_conversion_wrong(monkeypatch, mutate)
    model = _witness(factory)
    broken = converter.trace_and_convert(copy.deepcopy(model), shape, "cpu", packaging=packaging).flow
    assert probe_forward(broken, shape, "cpu", strict=False, dtype=torch.float64).ok, (
        "the zero-input structural probe alone does not see this mutation"
    )
    with pytest.raises(ConversionValueParityError) as excinfo:
        convert_torch_model(model, shape, 6, packaging=packaging)
    assert excinfo.value.first_diverging_node == node, str(excinfo.value)


def test_value_parity_refusal_reports_magnitudes_and_node(monkeypatch):
    """A wrong operand order (the D11 shape, made wrong on purpose): the refusal names the node and both magnitudes."""
    _make_conversion_wrong(monkeypatch, _swap_sub_operands)
    with pytest.raises(ConversionValueParityError) as excinfo:
        convert_torch_model(_witness(_TwoBranchSub), (12,), 6, packaging=MVM_PACKAGING)
    exc, message = excinfo.value, str(excinfo.value)
    assert exc.max_abs_error is not None and exc.max_abs_error > 0
    assert exc.reference_max is not None and exc.reference_max > 0
    assert exc.first_diverging_node == "sub"
    assert "max|err|" in message and "max|ref|" in message and "'sub'" in message


# ----------------------------------------------------------------------------- currently exact programs
class _LinBNLin(nn.Module):
    """Linear -> BN -> Linear: graph normalization fuses the pair (fold computed in the model's dtype)."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 16)
        self.bn = nn.BatchNorm1d(16)
        self.l2 = nn.Linear(16, 8)
        self.r = nn.ReLU()
        self.h = nn.Linear(8, 4)

    def forward(self, x):
        return self.h(self.r(self.l2(self.bn(self.l1(x)))))


class _ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.b1, self.r1 = nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU()
        self.c2, self.b2 = nn.Conv2d(8, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8)
        self.sc = nn.Conv2d(3, 8, 1, bias=False)
        self.r2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        y = self.b2(self.c2(self.r1(self.b1(self.c1(x)))))
        return self.fc(torch.flatten(self.pool(self.r2(y + self.sc(x))), 1))


class _ParamMatmul(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(32, 10) * 0.1)

    def forward(self, x):
        return torch.matmul(x, self.w).mean(dim=1)


class _MhaLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(32, 16)
        self.norm = nn.LayerNorm(16)
        self.attn = nn.MultiheadAttention(16, 2, batch_first=True)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        t = self.embed(x)
        n = self.norm(t)
        a, _ = self.attn(n, n, n)
        return self.head((t + a).mean(dim=1))


SMALL = (3, 8, 8)
EXACT_PROGRAMS: list[tuple[str, Factory, tuple[int, ...]]] = [
    ("op_relu", lambda: OpWitness(nn.ReLU(), spatial_after=8), SMALL),
    ("op_leakyrelu", lambda: OpWitness(nn.LeakyReLU(), spatial_after=8), SMALL),
    ("op_gelu", lambda: OpWitness(nn.GELU(), spatial_after=8), SMALL),
    ("op_silu", lambda: OpWitness(nn.SiLU(), spatial_after=8), SMALL),
    ("op_tanh", lambda: OpWitness(nn.Tanh(), spatial_after=8), SMALL),
    ("op_bn_after_conv", lambda: OpWitness(nn.BatchNorm2d(8), spatial_after=8), SMALL),
    ("op_groupnorm", lambda: OpWitness(nn.GroupNorm(2, 8), spatial_after=8), SMALL),
    ("op_avgpool", lambda: OpWitness(nn.AvgPool2d(2), spatial_after=4), SMALL),
    ("op_maxpool", lambda: OpWitness(nn.MaxPool2d(2), spatial_after=4), SMALL),
    ("op_adaptive_avg", lambda: OpWitness(nn.AdaptiveAvgPool2d(2), spatial_after=2), SMALL),
    ("op_dilated", lambda: OpWitness(nn.ReLU(), conv_kw={"dilation": 2, "padding": 2}, spatial_after=8), SMALL),
    ("op_conv_transpose", lambda: OpWitness(nn.ConvTranspose2d(8, 8, 2, stride=2), spatial_after=16), SMALL),
    ("op_upsample", lambda: OpWitness(nn.Upsample(scale_factor=2), spatial_after=16), SMALL),
    ("op_dropout", lambda: OpWitness(nn.Dropout(0.1), spatial_after=8), SMALL),
    ("linear_bn_eps_0.1", lambda: BnEpsWitness(1e-1, lin=True), (3, 16, 16)),
    ("linear_bn_linear_fused", _LinBNLin, (16,)),
    ("residual_block", _ResidualBlock, SMALL),
    ("param_matmul", _ParamMatmul, SSHAPE),
    ("mha_leaf", _MhaLeaf, SSHAPE),
]


@pytest.mark.parametrize("packaging_name", sorted(PACKAGINGS))
@pytest.mark.parametrize("name,factory,shape", EXACT_PROGRAMS, ids=[p[0] for p in EXACT_PROGRAMS])
def test_currently_exact_program_passes_the_probe(name, factory, shape, packaging_name):
    kind, value = _converted_outcome(_witness(factory), shape, PACKAGINGS[packaging_name])
    assert kind == "measured", f"an exact program was refused: {value}"
    assert value <= EXACT


@pytest.mark.parametrize("name,factory,shape", EXACT_PROGRAMS, ids=[p[0] for p in EXACT_PROGRAMS])
def test_currently_exact_program_passes_the_value_program_stage(name, factory, shape):
    kind, value = _value_program_outcome(_witness(factory), shape)
    assert kind == "measured", f"an exact program was refused: {value}"
    assert value <= EXACT


def test_fp32_model_converts_and_its_fp64_witness_is_exact():
    """The probe converts an fp64 witness of an fp32 model; the returned flow stays fp32."""
    torch.manual_seed(0)
    model = _LinBNLin().eval()
    flow = convert_torch_model(model, (16,), 4, packaging=MVM_PACKAGING)
    assert all(p.dtype == torch.float32 for p in flow.parameters() if p.is_floating_point())


# ----------------------------------------------------------------------------- stage 2 (the pipeline's value-domain conversion)
def _baseline_d9_gather(original):
    """The pre-fix Conv1d lowering: indices (C, L, k) read as (L, C*k) (the D9 defect)."""
    import numpy as np

    def gather(padded, kernel_size, stride, dilation, output_size):
        if len(kernel_size) != 1:
            return original(padded, kernel_size, stride, dilation, output_size)
        k, s, d, length = int(kernel_size[0]), int(stride[0]), int(dilation[0]), int(output_size[0])
        l_idx = (np.arange(length) * s)[:, None] + (np.arange(k) * d)[None, :]
        c_idx = np.arange(padded.shape[0])[:, None, None]
        lb, cb = np.broadcast_arrays(l_idx[None, :, :], c_idx)
        return padded[cb, lb].reshape(length, padded.shape[0] * k)

    return gather


class _Conv1dNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(4, 6, 3, padding=1)
        self.a = nn.ReLU()
        self.fc = nn.Linear(6 * 16, 3)

    def forward(self, x):
        return self.fc(self.a(self.c(x)).flatten(1))


def test_value_domain_conversion_refuses_a_wrong_lowering_at_the_value_program(monkeypatch):
    """D9 reinjected: the converted model is exact, only the IR lowering is wrong; stage 2 names the conv."""
    from mimarsinan.mapping.mappers import conv_helpers

    monkeypatch.setattr(conv_helpers, "gather_conv_patches", _baseline_d9_gather(conv_helpers.gather_conv_patches))
    model = _witness(_Conv1dNet)
    with pytest.raises(ConversionValueParityError) as excinfo:
        convert_torch_model(model, (4, 16), 3, packaging=MVM_PACKAGING, value_program_stage=True)
    assert excinfo.value.stage == "identity value program", str(excinfo.value)
    assert excinfo.value.first_diverging_node == "c", str(excinfo.value)
    assert excinfo.value.first_diverging_operator == "Conv1d"
    assert "'c' (Conv1d)" in str(excinfo.value)


def test_event_domain_conversion_does_not_lower_to_a_value_program(monkeypatch):
    from mimarsinan.mapping.mappers import conv_helpers

    monkeypatch.setattr(conv_helpers, "gather_conv_patches", _baseline_d9_gather(conv_helpers.gather_conv_patches))
    flow = convert_torch_model(_witness(_Conv1dNet), (4, 16), 3, packaging=SPIKING_PACKAGING,
                               value_program_stage=True)
    assert [v.stage for v in flow.conversion_certificate.stages] == ["converted model"]


def test_value_program_stage_passes_on_the_unmutated_lowering():
    flow = convert_torch_model(_witness(_Conv1dNet), (4, 16), 3, packaging=MVM_PACKAGING, value_program_stage=True)
    stages = flow.conversion_certificate.stages
    assert [v.stage for v in stages] == ["converted model", "identity value program"]
    assert all(v.ok and v.relative_error <= EXACT for v in stages)
