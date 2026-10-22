"""Conv conversion exactness: native fp64 model == converted model == IR value program, or a named refusal.

Pins three conversion-edge defects (TCAD significance review, Wave 0):

- D9: the Conv1d IR gather mixed channels and positions for C > 1 and
  L_out > 1 (value program 0.5 to 1.3 max-abs off the native model).
- D9b: a spiking bare Conv1d (no activation) became a host op without shapes
  and crashed IR mapping in the zeros-tensor shape probe.
- D2 (E273.R): a non-zero ``padding_mode`` on the perceptron conv path was
  converted as zero padding; representability now refuses it by name, beside
  the grouped-convolution refusal, while a host conv op (which runs the native
  module) keeps its padding mode exactly. String paddings resolve to their
  numeric zero padding ('valid', symmetric 'same') or are refused the same way.

Every model is built inside the test, converted in fp64 from a deep copy whose
BatchNorm running statistics are randomized, and compared on two seeded
batches of two at 1e-12.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.torch_mapping.converter import check_representability, convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.conv_mixin import UnsupportedConvPaddingError
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
    RepresentabilityError,
    RepresentabilityReport,
)
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

_TOL = 1e-12
_NUM_CLASSES = 10
_C_IN, _C_OUT, _LENGTH = 16, 8, 32

# name -> (kernel, stride, padding, dilation); groups 1 throughout.
_GEOMETRIES = {
    "k1p0": (1, 1, 0, 1),
    "k3p1": (3, 1, 1, 1),
    "k3p1_s2": (3, 2, 1, 1),
    "k3p2_d2": (3, 1, 2, 2),
}
_NON_ZERO_MODES = ("reflect", "replicate", "circular")


def _l_out(k, s, p, d, length=_LENGTH):
    return (length + 2 * p - d * (k - 1) - 1) // s + 1


class _Conv1dHead(nn.Module):
    """Conv1d(16 -> 8) [-> BN] [-> ReLU] -> flatten -> FC (under MVM the conv is on chip)."""

    def __init__(self, k, s, p, d, *, bn=False, act=True, padding_mode="zeros"):
        super().__init__()
        self.conv = nn.Conv1d(_C_IN, _C_OUT, k, stride=s, padding=p, dilation=d,
                              padding_mode=padding_mode)
        self.bn = nn.BatchNorm1d(_C_OUT) if bn else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        self.fc = nn.Linear(_C_OUT * _l_out(k, s, p, d), _NUM_CLASSES)

    def forward(self, x):
        return self.fc(torch.flatten(self.act(self.bn(self.conv(x))), 1))


class _EncodedConv1d(nn.Module):
    """1x1 Conv1d + ReLU (the spiking encoding layer), then the Conv1d under test + ReLU, then FC.

    Under the spiking contract the second conv is the on-chip Conv1d.
    """

    def __init__(self, k, s, p, d):
        super().__init__()
        self.encode = nn.Conv1d(_C_IN, _C_IN, 1)
        self.encode_act = nn.ReLU()
        self.conv = nn.Conv1d(_C_IN, _C_OUT, k, stride=s, padding=p, dilation=d)
        self.act = nn.ReLU()
        self.fc = nn.Linear(_C_OUT * _l_out(k, s, p, d), _NUM_CLASSES)

    def pre_activation(self, x):
        return self.conv(self.encode_act(self.encode(x)))

    def forward(self, x):
        return self.fc(torch.flatten(self.act(self.pre_activation(x)), 1))


class _CausalConv1dTwin(nn.Module):
    """Mamba-style short causal conv, groups 1 twin: Linear -> transpose -> Conv1d k4p3 -> slice -> SiLU -> mean."""

    def __init__(self, steps=16, features=32, width=64):
        super().__init__()
        self.steps = steps
        self.inp = nn.Linear(features, width)
        self.conv = nn.Conv1d(width, width, 4, padding=3)
        self.act = nn.SiLU()
        self.head = nn.Linear(width, _NUM_CLASSES)

    def forward(self, x):
        y = self.inp(x).transpose(1, 2)
        y = self.act(self.conv(y)[..., : self.steps])
        return self.head(y.mean(dim=2))


class _Conv2dHead(nn.Module):
    """Conv2d(3 -> 8, k3 p1) [-> ReLU] -> flatten -> FC on a 3 x 8 x 8 input."""

    def __init__(self, *, act=True, padding_mode="zeros"):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, padding_mode=padding_mode)
        self.act = nn.ReLU() if act else nn.Identity()
        self.fc = nn.Linear(8 * 8 * 8, _NUM_CLASSES)

    def forward(self, x):
        return self.fc(torch.flatten(self.act(self.conv(x)), 1))


def _randomize_bn(model, seed=7):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                n = m.num_features
                m.running_mean.copy_(torch.rand(n, generator=gen, dtype=torch.float64) * 0.4 - 0.2)
                m.running_var.copy_(torch.rand(n, generator=gen, dtype=torch.float64) * 1.5 + 0.5)
                m.weight.copy_(torch.rand(n, generator=gen, dtype=torch.float64) * 1.5 + 0.5)
                m.bias.copy_(torch.rand(n, generator=gen, dtype=torch.float64) * 0.4 - 0.2)


def _native(factory):
    torch.manual_seed(0)
    native = copy.deepcopy(factory().double().eval())
    _randomize_bn(native)
    return native


def _batches(shape, *, uniform=False):
    out = []
    for seed in (1, 2):
        gen = torch.Generator().manual_seed(seed)
        draw = torch.rand if uniform else torch.randn
        out.append(draw(2, *shape, dtype=torch.float64, generator=gen))
    return out


def _max_abs(got, want):
    return float((got.reshape(want.shape) - want).abs().max())


def _convert(native, shape, packaging):
    return convert_torch_model(native, shape, _NUM_CLASSES, packaging=packaging).eval()


def _value_program(flow):
    ir = map_model_to_ir(flow.get_mapper_repr())
    program = ValueHybridCoreFlow(
        build_identity_hybrid_mapping(ir_graph=ir), device="cpu", dtype=torch.float64,
    )
    return ir, program


def _assert_exact(native, flow, program, batches):
    with torch.no_grad():
        for x in batches:
            want = native(x)
            assert _max_abs(flow(x), want) <= _TOL, "converted model != native"
            if program is not None:
                err = _max_abs(program(x), want)
                assert err <= _TOL, f"value program != native: max-abs {err:.3e}"


def _conv_cores(ir, node_name="conv"):
    return [n for n in ir.nodes if isinstance(n, NeuralCore) and n.name.startswith(f"{node_name}_pos")]


class TestConv1dMvmExactness:
    """D9: the on-chip Conv1d (MVM) is exact for C > 1 and L_out > 1."""

    @pytest.mark.parametrize("act", [True, False], ids=["relu", "bare"])
    @pytest.mark.parametrize("bn", [False, True], ids=["nobn", "bn"])
    @pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
    def test_converted_model_and_value_program_exact(self, geometry, bn, act):
        k, s, p, d = _GEOMETRIES[geometry]
        native = _native(lambda: _Conv1dHead(k, s, p, d, bn=bn, act=act))
        flow = _convert(native, (_C_IN, _LENGTH), MVM_PACKAGING)
        ir, program = _value_program(flow)
        assert len(_conv_cores(ir)) == _l_out(k, s, p, d) > 1
        _assert_exact(native, flow, program, _batches((_C_IN, _LENGTH)))

    def test_causal_dense_twin_value_program_exact(self):
        native = _native(_CausalConv1dTwin)
        flow = _convert(native, (16, 32), MVM_PACKAGING)
        ir, program = _value_program(flow)
        assert len(_conv_cores(ir)) == 16 + 3
        _assert_exact(native, flow, program, _batches((16, 32)))


class TestConv1dSpikingExactness:
    """D9 under the spiking contract: the on-chip Conv1d cores compute the native conv.

    The conv under test has non-negative weights and a positive bias over
    non-negative inputs, so its ReLU never clips (asserted) and the affine
    value program evaluates the spiking IR's cores exactly.
    """

    @pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
    def test_on_chip_conv1d_value_program_exact(self, geometry):
        k, s, p, d = _GEOMETRIES[geometry]
        native = _native(lambda: _EncodedConv1d(k, s, p, d))
        with torch.no_grad():
            native.conv.weight.abs_()
            native.conv.bias.abs_().add_(0.01)
        batches = _batches((_C_IN, _LENGTH), uniform=True)
        with torch.no_grad():
            assert all(float(native.pre_activation(x).min()) > 0.0 for x in batches)

        flow = _convert(native, (_C_IN, _LENGTH), SPIKING_PACKAGING)
        ir, program = _value_program(flow)
        cores = _conv_cores(ir)
        assert len(cores) == _l_out(k, s, p, d) > 1
        assert all(core.activation_type is not None for core in cores)
        _assert_exact(native, flow, program, batches)


class TestSpikingBareConv1dHostOp:
    """D9b: a spiking Conv1d without activation is one host op over the whole (C, L) grid."""

    @pytest.mark.parametrize("bn", [False, True], ids=["nobn", "bn"])
    @pytest.mark.parametrize("geometry", ["k3p1", "k3p1_s2", "k3p2_d2"])
    def test_bare_conv1d_maps_and_is_exact(self, geometry, bn):
        k, s, p, d = _GEOMETRIES[geometry]
        native = _native(lambda: _Conv1dHead(k, s, p, d, bn=bn, act=False))
        flow = _convert(native, (_C_IN, _LENGTH), SPIKING_PACKAGING)
        ir, program = _value_program(flow)

        (conv_op,) = [n for n in ir.nodes if isinstance(n, ComputeOp) and n.name == "conv"]
        assert conv_op.input_shape == (_C_IN, _LENGTH)
        assert conv_op.output_shape == (_C_OUT, _l_out(k, s, p, d))
        assert not _conv_cores(ir)
        _assert_exact(native, flow, program, _batches((_C_IN, _LENGTH)))


class TestSpikingBareConvHostOpShapes:
    """Both conv host paths hand the host op the batch-stripped FX shapes of the whole grid (one rule)."""

    def test_bare_conv2d_host_op_carries_the_whole_grid(self):
        native = _native(lambda: _Conv2dHead(act=False))
        flow = _convert(native, (3, 8, 8), SPIKING_PACKAGING)
        ir, program = _value_program(flow)
        (conv_op,) = [n for n in ir.nodes if isinstance(n, ComputeOp) and n.name == "conv"]
        assert conv_op.input_shape == (3, 8, 8)
        assert conv_op.output_shape == (8, 8, 8)
        assert not _conv_cores(ir)
        _assert_exact(native, flow, program, _batches((3, 8, 8)))


_PERCEPTRON_PATHS = [
    pytest.param(MVM_PACKAGING, True, id="mvm-relu"),
    pytest.param(MVM_PACKAGING, False, id="mvm-bare"),
    pytest.param(SPIKING_PACKAGING, True, id="spiking-relu"),
]


def _padded_witness(kind, mode, act):
    if kind == "Conv1d":
        return (lambda: _Conv1dHead(3, 1, 1, 1, act=act, padding_mode=mode)), (_C_IN, _LENGTH)
    return (lambda: _Conv2dHead(act=act, padding_mode=mode)), (3, 8, 8)


def _unpadded_witness(kind, mode, act):
    if kind == "Conv1d":
        return lambda: _Conv1dHead(3, 1, 0, 1, act=act, padding_mode=mode)

    class _UnpaddedConv2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=0, padding_mode=mode)
            self.act = nn.ReLU() if act else nn.Identity()
            self.fc = nn.Linear(8 * 6 * 6, _NUM_CLASSES)

        def forward(self, x):
            return self.fc(torch.flatten(self.act(self.conv(x)), 1))

    return _UnpaddedConv2d


def _padding_refusal(report: RepresentabilityReport):
    (op,) = [op for op in report.unsupported_ops if op.node_name == "conv"]
    return op


class TestConvPaddingModeRefusal:
    """E273.R: the perceptron conv gather pads with zeros only; other modes are refused by name at representability."""

    @pytest.mark.parametrize("packaging,act", _PERCEPTRON_PATHS)
    @pytest.mark.parametrize("mode", _NON_ZERO_MODES)
    @pytest.mark.parametrize("kind", ["Conv1d", "Conv2d"])
    def test_non_zero_mode_on_perceptron_path_is_refused_by_name(self, kind, mode, packaging, act):
        factory, shape = _padded_witness(kind, mode, act)
        report = check_representability(_native(factory), shape, packaging=packaging)
        assert not report.is_representable
        op = _padding_refusal(report)
        assert (op.node_name, op.op_type, op.module_type) == ("conv", "call_module", kind)
        assert repr(mode) in op.reason and "zero padding" in op.reason
        with pytest.raises(RepresentabilityError) as info:
            _convert(_native(factory), shape, packaging)
        message = str(info.value)
        assert "conv" in message and kind in message and repr(mode) in message

    @pytest.mark.parametrize("packaging,act", _PERCEPTRON_PATHS)
    @pytest.mark.parametrize("kind", ["Conv1d", "Conv2d"])
    def test_zeros_mode_converts_exactly(self, kind, packaging, act):
        factory, shape = _padded_witness(kind, "zeros", act)
        native = _native(factory)
        assert check_representability(native, shape, packaging=packaging).is_representable
        flow = _convert(native, shape, packaging)
        program = _value_program(flow)[1] if packaging is MVM_PACKAGING else None
        _assert_exact(native, flow, program, _batches(shape))

    @pytest.mark.parametrize("mode", _NON_ZERO_MODES)
    @pytest.mark.parametrize("kind", ["Conv1d", "Conv2d"])
    def test_host_conv_keeps_its_padding_mode_exactly(self, kind, mode):
        factory, shape = _padded_witness(kind, mode, act=False)
        native = _native(factory)
        assert check_representability(native, shape, packaging=SPIKING_PACKAGING).is_representable
        flow = _convert(native, shape, SPIKING_PACKAGING)
        ir, program = _value_program(flow)
        assert [n.name for n in ir.nodes if isinstance(n, ComputeOp)][0] == "conv"
        _assert_exact(native, flow, program, _batches(shape))

    @pytest.mark.parametrize("packaging,act", _PERCEPTRON_PATHS)
    @pytest.mark.parametrize("mode", _NON_ZERO_MODES)
    @pytest.mark.parametrize("kind", ["Conv1d", "Conv2d"])
    def test_a_mode_without_padding_converts_exactly_as_zero_padding_0(self, kind, mode, packaging, act):
        """With no padding the mode has no effect, so the perceptron conv path keeps it (exact on w0-base too)."""
        native = _native(_unpadded_witness(kind, mode, act))
        shape = (_C_IN, _LENGTH) if kind == "Conv1d" else (3, 8, 8)
        assert check_representability(native, shape, packaging=packaging).is_representable
        flow = _convert(native, shape, packaging)
        mappers = flow.get_mapper_repr().execution_order()
        (conv,) = [m for m in mappers if type(m).__name__ == f"{kind[:-1]}DPerceptronMapper"]
        assert conv.padding == (0 if kind == "Conv1d" else (0, 0))
        program = _value_program(flow)[1] if packaging is MVM_PACKAGING else None
        _assert_exact(native, flow, program, _batches(shape))

    @pytest.mark.parametrize("kind", ["Conv1d", "Conv2d"])
    def test_the_converter_keeps_the_padding_invariant(self, kind):
        """A report that skipped the analyzer cannot slip a non-zero mode through: the converter refuses it too."""
        factory, shape = _padded_witness(kind, "circular", act=True)
        gm = normalize_fx_graph(trace_model(_native(factory), shape))
        report = RepresentabilityAnalyzer(gm, packaging=MVM_PACKAGING).analyze()
        report.unsupported_ops.clear()
        report.is_representable = True
        with pytest.raises(UnsupportedConvPaddingError) as info:
            MapperGraphConverter(gm, shape, packaging=MVM_PACKAGING).convert(report)
        assert (info.value.node_name, info.value.operator, info.value.padding_mode) == ("conv", kind, "circular")


def _string_padded(kind, padding, kernel):
    conv = (nn.Conv1d(2, 4, kernel, padding=padding) if kind == "Conv1d"
            else nn.Conv2d(2, 4, kernel, padding=padding))
    shape = (2, 9) if kind == "Conv1d" else (2, 5, 5)
    with torch.no_grad():
        out = conv(torch.zeros(1, *shape))
    model = nn.Sequential(conv, nn.ReLU(), nn.Flatten(), nn.Linear(out.numel(), _NUM_CLASSES))
    return model.double().eval(), shape


class TestStringPadding:
    """'valid' is zero padding 0; 'same' is zero padding d (k - 1) / 2 per side when that total is even."""

    @pytest.mark.parametrize("packaging", [MVM_PACKAGING, SPIKING_PACKAGING], ids=["mvm", "spiking"])
    @pytest.mark.parametrize("padding,kernel", [("valid", 3), ("same", 3), ("same", 5)])
    @pytest.mark.parametrize("kind", ["Conv1d", "Conv2d"])
    def test_expressible_string_padding_converts_exactly(self, kind, padding, kernel, packaging):
        native, shape = _string_padded(kind, padding, kernel)
        flow = _convert(native, shape, packaging)
        program = _value_program(flow)[1] if packaging is MVM_PACKAGING else None
        _assert_exact(native, flow, program, _batches(shape))

    @pytest.mark.parametrize("kind", ["Conv1d", "Conv2d"])
    def test_asymmetric_same_padding_is_refused_by_name(self, kind):
        native, shape = _string_padded(kind, "same", 4)
        report = check_representability(native, shape, packaging=MVM_PACKAGING)
        op = [op for op in report.unsupported_ops if op.module_type == kind]
        assert op and "'same'" in op[0].reason and "asymmetric" in op[0].reason
        with pytest.raises(RepresentabilityError):
            _convert(native, shape, MVM_PACKAGING)
