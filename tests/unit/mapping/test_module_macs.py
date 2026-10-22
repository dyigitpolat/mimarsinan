"""The forward-MAC SSOT (``mapping/support/module_macs.py``): one rule set, read by both MAC counters.

``count_host_ops`` (the mapped IR's host ops) and ``estimate_onchip_fraction(metric='macs')`` (the converted flow's
forward) count every invocation at its shapes, every output position included: an on-chip convolution at each output
position (its mapper drives the perceptron through ``F.conv2d``, so no hook on the perceptron fires), a host op split
per column at each column, a multi-input op (the attention leaf) from its operands' shapes.
"""

from __future__ import annotations

import ast
import contextlib
import io
import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.mapping.verification.onchip_majority import count_host_ops, module_macs
from mimarsinan.torch_mapping.converter import trace_and_convert

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


# ----------------------------------------------------------------------------- rules
class TestRules:
    def test_linear_counts_every_leading_position(self):
        assert module_macs(nn.Linear(8, 4), (1, 5, 8), (1, 5, 4)) == 5 * 8 * 4

    def test_conv_counts_every_output_position_with_groups(self):
        conv = nn.Conv2d(8, 12, 3, padding=1, groups=4)
        assert module_macs(conv, (1, 8, 6, 6), (1, 12, 6, 6)) == 12 * 36 * (8 // 4) * 9

    def test_transposed_conv_is_its_scatter_form_with_groups(self):
        conv = nn.ConvTranspose2d(8, 6, 2, stride=2, groups=2)
        # every input element times every kernel weight of its group: Cin * (Cout / g) * k * k per input position
        assert module_macs(conv, (1, 8, 5, 5), (1, 6, 10, 10)) == 8 * 25 * (6 // 2) * 4

    def test_perceptron_conv_counts_its_owners_output_positions(self):
        from mimarsinan.mapping.mapping_utils import Conv2DPerceptronMapper
        mapper = Conv2DPerceptronMapper(None, in_channels=3, out_channels=8, kernel_size=3, padding=1)
        assert module_macs(mapper.perceptron, (1, 3, 10, 10), (1, 8, 10, 10)) == 8 * 100 * 27

    def test_self_attention_leaf(self):
        e, L = 16, 7
        mha = nn.MultiheadAttention(e, 2, batch_first=True)
        assert module_macs(mha, (1, L, e), (1, L, e)) == 4 * e * e * L + 2 * L * L * e

    def test_cross_attention_reads_every_operand_shape(self):
        e, lq, lk, kd = 16, 5, 9, 6
        mha = nn.MultiheadAttention(e, 2, kdim=kd, vdim=kd, batch_first=True)
        shapes = [(1, lq, e), (1, lk, kd), (1, lk, kd)]
        want = lq * e * e + lk * e * kd + lk * e * kd + lq * e * e + 2 * lq * lk * e
        assert module_macs(mha, None, (1, lq, e), shapes) == want

    @pytest.mark.parametrize("cls,gates", [(nn.LSTM, 4), (nn.GRU, 3), (nn.RNN, 1)])
    def test_recurrent_leaf_per_layer_and_direction(self, cls, gates):
        t, f, h = 6, 5, 7
        rnn = cls(f, h, num_layers=2, bidirectional=True, batch_first=True)
        want = t * gates * h * (f + h) * 2 + t * gates * h * (2 * h + h) * 2
        assert module_macs(rnn, (1, t, f), (1, t, 2 * h)) == want

    def test_transformer_encoder_leaves_count_attention_and_both_linears(self):
        layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, batch_first=True)
        per_layer = 4 * 8 * 8 * 5 + 2 * 5 * 5 * 8 + 2 * 5 * 8 * 16
        assert module_macs(layer, (1, 5, 8), (1, 5, 8)) == per_layer
        encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        assert module_macs(encoder, (1, 5, 8), (1, 5, 8)) == 3 * per_layer

    def test_a_declared_count_wins(self):
        class Declared(nn.Module):
            def forward_macs(self):
                return 1234
        assert module_macs(Declared(), None, None) == 1234

    def test_the_converters_host_weight_package_counts_its_weight_op(self):
        package = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
        assert module_macs(package, (1, 3, 8, 8), (1, 4, 6, 6)) == 4 * 36 * 3 * 9

    def test_elementwise_ops_carry_none(self):
        assert module_macs(nn.ReLU(), (1, 8), (1, 8)) == 0


# ----------------------------------------------------------------------------- both counters
def _ir(flow):
    with contextlib.redirect_stdout(io.StringIO()):
        return IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
                         allow_coalescing=False, hardware_bias=False).map(flow.get_mapper_repr())


def _flow(model, shape, packaging):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        return trace_and_convert(model.eval(), shape, packaging=packaging).flow


def _counters(model, shape, packaging):
    flow = _flow(model, shape, packaging)
    est = estimate_onchip_fraction(flow, shape, 10, metric="macs")
    return count_host_ops(_ir(flow)), est


class _Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()
        self.down = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.b2(self.c2(self.relu(self.b1(self.c1(x)))))
        return self.relu(y + (x if self.down is None else self.down(x)))


def _cifar_resnet18(width=0.25, classes=100):
    widths = [max(8, int(c * width / 8 + 0.5) * 8) for c in (64, 128, 256, 512)]
    blocks, cin = [], widths[0]
    for stage, c in enumerate(widths):
        for b in range(2):
            blocks.append(_Basic(cin, c, 2 if stage > 0 and b == 0 else 1))
            cin = c
    return nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(),
                         *blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, classes))


def _analytic_macs(model, shape):
    """Weight-op MACs of the native model from its own forward (an independent count, not the SSOT)."""
    total = []

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            total.append(output[0].numel() * (module.in_channels // module.groups) * k)
        elif isinstance(module, nn.Linear):
            total.append(output[0].numel() * module.in_features)
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(1, *shape))
    for handle in handles:
        handle.remove()
    return sum(total)


@pytest.mark.parametrize("packaging", [MVM_PACKAGING, SPIKING_PACKAGING], ids=["mvm", "spiking"])
def test_resnet18_w_quarter_total_is_the_analytic_count(packaging):
    """CIFAR ResNet-18 at width 1/4: every conv at every output position (on the base, 710,576 of 35,058,176)."""
    torch.manual_seed(0)
    model = _cifar_resnet18().eval()
    want = _analytic_macs(model, (3, 32, 32))
    host, est = _counters(model, (3, 32, 32), packaging)
    assert want == 35_058_176
    assert abs(est.total - want) <= 0.01 * want
    assert host == est.host


class _ConvEncoderNet(nn.Module):
    """A subsumed conv encoder (host), a host conv before a residual add, an on-chip conv, a host readout."""

    def __init__(self):
        super().__init__()
        self.c0, self.r0 = nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()
        self.c1 = nn.Conv2d(8, 8, 3, padding=1)
        self.c2, self.r2 = nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        self.head = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        y = self.r0(self.c0(x))
        y = self.r2(self.c2(self.c1(y) + y))
        return self.head(torch.flatten(y, 1))


class _TokenLinear(nn.Module):
    """A bare Linear over tokens: a host op split per token column under the spiking contract."""

    def __init__(self):
        super().__init__()
        self.a, self.act = nn.Linear(6, 6), nn.ReLU()
        self.b = nn.Linear(6, 4)

    def forward(self, x):
        return self.b(self.act(self.a(x))).mean(dim=1)


class _AttentionLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp, self.act = nn.Linear(8, 16), nn.ReLU()
        self.mha = nn.MultiheadAttention(16, 2, batch_first=True)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        y = self.act(self.inp(x))
        return self.head(self.mha(y, y, y)[0].mean(dim=1))


class _RecurrentLeaf(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.rnn = cell(8, 12, batch_first=True)
        self.head = nn.Linear(12, 10)

    def forward(self, x):
        return self.head(self.rnn(x)[0].mean(dim=1))


@pytest.mark.parametrize("model,shape,packaging,host_macs", [
    (_ConvEncoderNet, (3, 8, 8), SPIKING_PACKAGING, 8 * 64 * 27 + 8 * 64 * 72 + 512 * 10),
    (_TokenLinear, (5, 6), SPIKING_PACKAGING, 5 * 6 * 6 + 5 * 6 * 4),
    (_AttentionLeaf, (7, 8), MVM_PACKAGING, 4 * 16 * 16 * 7 + 2 * 7 * 7 * 16),
], ids=["conv_encoder", "token_linear", "attention_leaf"])
def test_both_counters_count_every_host_position(model, shape, packaging, host_macs):
    host, est = _counters(model(), shape, packaging)
    assert host == est.host == host_macs


@pytest.mark.parametrize("cell,gates", [(nn.LSTM, 4), (nn.GRU, 3)])
def test_both_counters_count_a_recurrent_leaf(cell, gates):
    host, est = _counters(_RecurrentLeaf(cell), (6, 8), MVM_PACKAGING)
    assert host == est.host == 6 * gates * 12 * (8 + 12)


def test_a_module_in_two_ops_counts_both_invocations():
    shared = nn.Linear(10, 4)
    sources = np.array([IRSource(node_id=-2, index=i) for i in range(10)], dtype=object)
    ops = [ComputeOp(id=i, name=f"op{i}", input_sources=sources, op_type="Linear", params={"module": shared},
                     input_shape=(10,), output_shape=(4,)) for i in range(2)]
    graph = IRGraph(nodes=ops, output_sources=np.array([IRSource(node_id=1, index=0)], dtype=object))
    assert count_host_ops(graph) == 80


# ----------------------------------------------------------------------------- one rule set
_EXEMPT = {("mapping/crossbar_utilization.py", "macs"), ("deployment_record/platform_physics/host_calibration.py",
                                                          "macs_of")}
_SIZES = frozenset({"in_features", "out_features", "in_channels", "out_channels", "kernel_size", "groups",
                    "hidden_size", "embed_dim"})
"""A module's size attributes: a MAC count multiplies them."""


def _chain(node: ast.AST) -> tuple:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return tuple(reversed([*parts, node.id])) if isinstance(node, ast.Name) else ()


def _torch_nn_references(tree: ast.Module):
    """A predicate on an expression node: it names a torch.nn class (``nn.X``, ``torch.nn.X``, a class imported from
    ``torch.nn`` under any name, or a module-level tuple of them)."""
    modules, classes = {"nn"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[:2] == ["torch", "nn"]:
            classes.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "nn")
        elif isinstance(node, ast.Import):
            modules.update(alias.asname for alias in node.names if alias.name == "torch.nn" and alias.asname)

    def direct(node: ast.AST) -> bool:
        chain = _chain(node)
        return (len(chain) == 1 and chain[0] in classes) or (len(chain) >= 2 and (
            chain[0] in modules or chain[:2] == ("torch", "nn")))

    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Tuple, ast.List, ast.Set)) \
                and any(direct(e) for e in ast.walk(node.value)):
            classes.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return direct


def _type_check(node: ast.AST, names_class) -> bool:
    """``isinstance``/``issubclass`` against a torch.nn class, or ``type(m) is / == / in`` one."""
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("isinstance", "issubclass") \
            and len(node.args) == 2:
        return any(names_class(n) for n in ast.walk(node.args[1]))
    if isinstance(node, ast.Compare):
        sides = [node.left, *node.comparators]
        return any(isinstance(s, ast.Call) and getattr(s.func, "id", None) == "type" for s in sides) and any(
            names_class(n) for s in sides for n in ast.walk(s))
    return False


def _size_product(node: ast.AST) -> bool:
    product = (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)) or (
        isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "prod")
    return product and any(isinstance(n, ast.Attribute) and n.attr in _SIZES for n in ast.walk(node))


def _callee(node: ast.AST):
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    owner = node.func.value if isinstance(node.func, ast.Attribute) else None
    return node.func.attr if isinstance(owner, ast.Name) and owner.id in ("self", "cls") else None


def mac_dispatchers(source: str) -> list:
    """Every function, whatever its name, that type-checks a torch.nn class and multiplies size attributes, itself
    or through a function of its module that does (a counter split into a dispatcher and its rules)."""
    tree = ast.parse(source)
    names_class = _torch_nn_references(tree)
    functions = [f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
    multiplying = {f.name for f in functions if any(_size_product(n) for n in ast.walk(f))}
    return [f.name for f in functions if any(_type_check(n, names_class) for n in ast.walk(f)) and (
        f.name in multiplying or any(_callee(n) in multiplying for n in ast.walk(f)))]


def test_no_function_outside_the_ssot_dispatches_on_module_type_to_count_macs():
    found = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel == "mapping/support/module_macs.py":
            continue
        found += [f"{rel}:{name}" for name in mac_dispatchers(path.read_text()) if (rel, name) not in _EXEMPT]
    assert found == []


EVADING = {
    "type_is_torch_nn": """
import torch
def _flops(m):
    if type(m) is torch.nn.Conv2d:
        return m.in_channels * m.out_channels
    return 0
""",
    "imported_class": """
from torch.nn import Linear as Dense
def _ops_of(m):
    return m.in_features * m.out_features if isinstance(m, Dense) else 0
""",
    "module_tuple_and_split_rule": """
import math
import torch.nn as nn
_WEIGHTED = (nn.Linear, nn.Conv2d)
def _rule(m):
    return math.prod(m.kernel_size) * m.out_channels
def _count(m):
    return _rule(m) if isinstance(m, _WEIGHTED) else 0
""",
    "type_in_tuple": """
from torch import nn
class Census:
    def _weight(self, m):
        return m.groups * m.out_channels
    def visit(self, m):
        return self._weight(m) if type(m) in (nn.Conv1d, nn.Conv2d) else 0
""",
}


@pytest.mark.parametrize("key", sorted(EVADING))
def test_the_rule_set_guard_finds_a_counter_whatever_its_name_and_dispatch_form(key):
    assert mac_dispatchers(EVADING[key]), key


def test_the_rule_set_guard_finds_the_ssot_itself():
    assert mac_dispatchers((SRC / "mapping/support/module_macs.py").read_text()) == ["module_macs"]
