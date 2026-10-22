"""FX literal coercion helpers + converter hardening around dynamic/unsupported FX arguments."""

import pytest
import torch
import torch.nn as nn
import torch.fx as fx

from mimarsinan.torch_mapping.fx_shape_utils import fx_literal_int, node_target_str
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import ProgramPurity, program_purity
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityError, RepresentabilityReport
from mimarsinan.torch_mapping.converter_handlers.batchnorm_lowering import copy_batchnorm_state


def _graph_with(builder):
    g = fx.Graph()
    nodes = builder(g)
    gm = fx.GraphModule(nn.Module(), g)
    return gm, nodes


# ── fx_literal_int ─────────────────────────────────────────────────────


def test_fx_literal_int_coerces_numeric_literals():
    assert fx_literal_int(3) == 3
    assert fx_literal_int(3.0) == 3
    assert fx_literal_int(True) == 1
    assert fx_literal_int("5") == 5


def test_fx_literal_int_rejects_fx_node():
    g = fx.Graph()
    x = g.placeholder("x")
    with pytest.raises(TypeError):
        fx_literal_int(x)


# ── node_target_str ────────────────────────────────────────────────────


def test_node_target_str_returns_string_targets():
    g = fx.Graph()
    x = g.placeholder("x")
    m = g.call_module("sub", (x,))
    meth = g.call_method("flatten", (x, 1))
    assert node_target_str(m) == "sub"
    assert node_target_str(meth) == "flatten"


def test_node_target_str_rejects_function_target():
    g = fx.Graph()
    x = g.placeholder("x")
    fn = g.call_function(torch.flatten, (x, 1))
    with pytest.raises(TypeError):
        node_target_str(fn)


# ── view with dynamic (Node) dims and no shape metadata ────────────────


def test_view_with_dynamic_dim_literal_falls_back_to_source():
    """Review round 9: the hand-built graph carries no shape metadata, so its receivers' run-time types are unknown and
    the program is impure by default (its view the host op of ``Tensor.view``); declared pure, the view keeps the
    per-sample fallback this test pins."""
    g = fx.Graph()
    x = g.placeholder("x")
    s = g.call_method("size", (x, 0))
    v = g.call_method("view", (x, (2, s)))
    g.output(v)
    gm = fx.GraphModule(nn.Module(), g)

    converter = MapperGraphConverter(gm, (4,), purity=ProgramPurity())
    converter._handle_placeholder(x)
    converter._handle_call_method(s)
    converter._handle_call_method(v)

    assert converter._node_to_mapper[v] is converter._node_to_mapper[x]
    assert program_purity(gm).impurity.node == "size"


# ── output node that does not resolve to a mapper ──────────────────────


def test_output_resolving_to_non_node_raises_runtime_error():
    g = fx.Graph()
    g.placeholder("x")
    g.output((3,))
    gm = fx.GraphModule(nn.Module(), g)

    converter = MapperGraphConverter(gm, (4,))
    report = RepresentabilityReport(is_representable=True)
    with pytest.raises(RuntimeError):
        converter.convert(report)


# ── string padding on perceptron-converted convolutions ────────────────
# 'same' with an odd kernel is symmetric zero padding (converted exactly); an
# even kernel needs asymmetric padding, which the perceptron conv gather cannot
# express, so representability refuses it by name.


class _Conv2dSamePadReLU(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 8 * 8, 4)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


class _Conv1dSamePadReLU(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(2, 4, kernel_size=kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 16, 4)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


class _Conv2dSamePadNoAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding="same")
        self.fc = nn.Linear(4 * 8 * 8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


def test_conv2d_asymmetric_string_padding_with_activation_is_refused():
    with pytest.raises(RepresentabilityError, match="padding 'same'"):
        convert_torch_model(_Conv2dSamePadReLU(kernel_size=4).eval(), (1, 8, 8), 4)


def test_conv1d_asymmetric_string_padding_with_activation_is_refused():
    with pytest.raises(RepresentabilityError, match="padding 'same'"):
        convert_torch_model(_Conv1dSamePadReLU(kernel_size=4).eval(), (2, 16), 4)


@pytest.mark.parametrize("model,shape", [(_Conv2dSamePadReLU, (1, 8, 8)), (_Conv1dSamePadReLU, (2, 16))])
def test_symmetric_string_padding_with_activation_converts_exactly(model, shape):
    native = model().double().eval()
    flow = convert_torch_model(native, shape, 4)
    x = torch.randn(3, *shape, dtype=torch.float64)
    with torch.no_grad():
        assert float((flow(x) - native(x)).abs().max()) <= 1e-12


def test_conv2d_string_padding_without_activation_still_converts():
    model = _Conv2dSamePadNoAct().eval()
    flow = convert_torch_model(model, (1, 8, 8), 4)
    x = torch.randn(2, 1, 8, 8)
    with torch.no_grad():
        expected = model(x)
        actual = flow(x)
    assert torch.allclose(expected, actual, atol=1e-5)


# ── copy_batchnorm_state ────────────────────────────────────────────────────


def _filled_bn(bn_cls, n):
    bn = bn_cls(n)
    with torch.no_grad():
        bn.weight.fill_(2.0)
        bn.bias.fill_(-1.0)
        bn.running_mean.fill_(0.5)
        bn.running_var.fill_(4.0)
        bn.num_batches_tracked.fill_(7)
    return bn


def test_copy_bn_params_copies_affine_and_running_stats():
    src = _filled_bn(nn.BatchNorm2d, 4)
    dst = nn.BatchNorm1d(4)
    copy_batchnorm_state(dst, src)
    assert torch.equal(dst.weight, src.weight)
    assert torch.equal(dst.bias, src.bias)
    assert torch.equal(dst.running_mean, src.running_mean)
    assert torch.equal(dst.running_var, src.running_var)
    assert int(dst.num_batches_tracked) == 7


def test_copy_bn_params_ignores_identity_destination():
    src = _filled_bn(nn.BatchNorm1d, 4)
    copy_batchnorm_state(nn.Identity(), src)
