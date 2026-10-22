"""D13: call-site weight sharing is refused by name until work item N7 implements it."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.torch_mapping.converter import check_representability, convert_torch_model
from mimarsinan.torch_mapping.fx_shape_utils import node_state, node_target_str, shared_state_sites
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.representability_analyzer import (
    CALL_SITE_WEIGHT_SHARING_REASON,
    RepresentabilityError,
)

PACKAGINGS = {"mvm": MVM_PACKAGING, "spiking": SPIKING_PACKAGING}


class UnrolledElman(nn.Module):
    """The recorded D13 witness: h_t = ReLU(W_x x_t + W_h h_{t-1}) unrolled over T steps, weights shared."""

    def __init__(self, hidden=32, steps=16, features=32):
        super().__init__()
        self.wx = nn.Linear(features, hidden)
        self.wh = nn.Linear(hidden, hidden, bias=False)
        self.act = nn.ReLU()
        self.steps = steps
        self.head = nn.Linear(hidden, 10)

    def forward(self, x):
        h = self.act(self.wx(x[:, 0]))
        for t in range(1, self.steps):
            h = self.act(self.wx(x[:, t]) + self.wh(h))
        return self.head(h)


class TiedEncoderDecoder(nn.Module):
    """Two distinct modules holding one Parameter (weight tying)."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(8, 8)
        self.dec = nn.Linear(8, 8)
        self.dec.weight = self.enc.weight

    def forward(self, x):
        return self.dec(torch.relu(self.enc(x)))


class ReusedBlock(nn.Module):
    """A parametric block applied twice (FX inlines it, so its leaves have two call sites)."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        return self.head(self.block(self.block(x)))


class ReusedStatelessModules(nn.Module):
    """One ReLU and one MaxPool module called at several sites: no state, nothing to tie."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3, padding=1)
        self.c2 = nn.Conv2d(4, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4 * 2 * 2, 3)

    def forward(self, x):
        y = self.pool(self.relu(self.c1(x)))
        y = self.pool(self.relu(self.c2(y)))
        return self.fc(torch.flatten(y, 1))


def _sharing_ops(report):
    return [op for op in report.unsupported_ops if CALL_SITE_WEIGHT_SHARING_REASON in (op.reason or "")]


@pytest.mark.parametrize("packaging_name", sorted(PACKAGINGS))
def test_unrolled_rnn_is_refused_by_name(packaging_name):
    packaging = PACKAGINGS[packaging_name]
    with pytest.raises(RepresentabilityError, match=CALL_SITE_WEIGHT_SHARING_REASON) as excinfo:
        convert_torch_model(UnrolledElman().eval(), (16, 32), 10, packaging=packaging)
    ops = _sharing_ops(excinfo.value.report)
    assert sorted(op.module_type for op in ops) == ["Linear", "Linear"]
    by_node = {op.node_name: op for op in ops}
    assert set(by_node) == {"wx", "wh"}
    assert "'wx'" in (by_node["wx"].reason or "") and "16 FX call sites" in (by_node["wx"].reason or "")
    assert "15 FX call sites" in (by_node["wh"].reason or "")
    assert "N7" in (by_node["wx"].reason or "")


def test_check_representability_reports_the_refusal():
    report = check_representability(UnrolledElman().eval(), (16, 32))
    assert not report.is_representable
    assert len(_sharing_ops(report)) == 2


def test_tied_parameter_across_modules_is_refused():
    with pytest.raises(RepresentabilityError, match=CALL_SITE_WEIGHT_SHARING_REASON) as excinfo:
        convert_torch_model(TiedEncoderDecoder().eval(), (8,), 8, packaging=MVM_PACKAGING)
    ops = _sharing_ops(excinfo.value.report)
    assert {op.node_name for op in ops} == {"enc", "dec"}
    assert all("enc, dec" in (op.reason or "") for op in ops)


def test_reused_parametric_block_is_refused():
    report = check_representability(ReusedBlock().eval(), (8,), packaging=MVM_PACKAGING)
    ops = _sharing_ops(report)
    assert [op.node_name for op in ops] == ["block_0"]
    assert "2 FX call sites" in (ops[0].reason or "")


@pytest.mark.parametrize("packaging_name", sorted(PACKAGINGS))
def test_reused_stateless_modules_are_not_refused(packaging_name):
    packaging = PACKAGINGS[packaging_name]
    model = ReusedStatelessModules().eval()
    report = check_representability(model, (3, 8, 8), packaging=packaging)
    assert report.is_representable and not _sharing_ops(report)
    flow = convert_torch_model(model, (3, 8, 8), 3, packaging=packaging)
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        assert torch.allclose(flow.eval()(x), model(x), atol=1e-6)


def test_models_without_reuse_report_identically():
    """The sharing census adds nothing to a model whose state is reached from one call site each."""
    model = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 3)).eval()
    report = check_representability(model, (6,))
    assert report.is_representable
    assert report.unsupported_ops == []
    assert [op.node_name for op in report.supported_ops] == ["_0", "_1", "_2"]


class SharedOnFusableChain(nn.Module):
    """A Linear-BN-Linear chain graph normalization would fuse, one of whose members another call site also reads."""

    def __init__(self, shared: str):
        super().__init__()
        self.shared = shared
        self.l1, self.bn, self.l2 = nn.Linear(16, 16), nn.BatchNorm1d(16), nn.Linear(16, 16)
        self.r, self.fc = nn.ReLU(), nn.Linear(16, 5)

    def forward(self, x):
        other = {"l1": self.l1, "bn": self.bn, "l2": self.l2}[self.shared]
        return self.fc(self.l2(self.bn(self.l1(x))) + self.r(other(2 * x)))


FUSABLE_CHAIN_SHARES = ["l1", "bn", "l2"]


@pytest.mark.parametrize("shared", FUSABLE_CHAIN_SHARES)
def test_graph_normalization_never_fuses_state_another_call_site_reads(shared):
    """Fusion would consume one of the call sites and hide the share from the census."""
    model = SharedOnFusableChain(shared).double().eval()
    gm = normalize_fx_graph(trace_model(model, (16,)))
    modules = dict(gm.named_modules())
    targets = [node_target_str(n) for n in gm.graph.nodes if n.op == "call_module"]
    assert targets.count(shared) == 2 and isinstance(modules[shared], type(getattr(model, shared)))
    assert not any(t.endswith("_fused") for t in targets)
    x = torch.randn(3, 16, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(gm(x), model(x))


@pytest.mark.parametrize("packaging_name", sorted(PACKAGINGS))
@pytest.mark.parametrize("shared", FUSABLE_CHAIN_SHARES)
def test_a_share_on_a_fusable_chain_is_refused_by_name(shared, packaging_name):
    packaging = PACKAGINGS[packaging_name]
    report = check_representability(SharedOnFusableChain(shared).eval(), (16,), packaging=packaging)
    ops = _sharing_ops(report)
    assert [op.node_name for op in ops] == [shared]
    assert "2 FX call sites" in (ops[0].reason or "")
    with pytest.raises(RepresentabilityError, match=CALL_SITE_WEIGHT_SHARING_REASON):
        convert_torch_model(SharedOnFusableChain(shared).eval(), (16,), 5, packaging=packaging)


def test_an_unshared_fusable_chain_still_fuses():
    model = nn.Sequential(nn.Linear(16, 16), nn.BatchNorm1d(16), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    gm = normalize_fx_graph(trace_model(model.double().eval(), (16,)))
    modules = dict(gm.named_modules())
    linears = [n for n in gm.graph.nodes if n.op == "call_module" and isinstance(modules[node_target_str(n)], nn.Linear)]
    assert len(linears) == 2


class _ParamPlusActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(4))
        self.l = nn.Linear(4, 4)

    def forward(self, x):
        return self.l(x) + self.shift


class TestCallSiteStateCensus:
    def test_buffers_are_not_call_site_state(self):
        gm = trace_model(ReusedConstantShift().eval(), (6,))
        modules = dict(gm.named_modules())
        attrs = [n for n in gm.graph.nodes if n.op == "get_attr"]
        assert attrs and all(node_state(n, gm, modules) == [] for n in attrs)
        assert shared_state_sites(gm, modules) == []

    def test_a_batchnorm_reads_its_parameters_not_its_statistics(self):
        gm = trace_model(nn.Sequential(nn.BatchNorm1d(3)).eval(), (3,))
        modules = dict(gm.named_modules())
        (bn,) = [n for n in gm.graph.nodes if n.op == "call_module"]
        module = modules[node_target_str(bn)]
        assert {id(t) for t in node_state(bn, gm, modules)} == {id(module.weight), id(module.bias)}

    def test_node_state_reads_module_and_attribute_state(self):
        gm = trace_model(_ParamPlusActivation().eval(), (4,))
        modules = dict(gm.named_modules())
        by_name = {n.name: n for n in gm.graph.nodes}
        assert {id(t) for t in node_state(by_name["l"], gm, modules)} == {id(gm.l.weight), id(gm.l.bias)}
        assert [id(t) for t in node_state(by_name["shift"], gm, modules)] == [id(gm.shift)]
        assert node_state(by_name["add"], gm, modules) == []

    def test_shared_state_sites_lists_every_site_in_graph_order(self):
        gm = trace_model(UnrolledElman(steps=3).eval(), (3, 32))
        shared = shared_state_sites(gm, dict(gm.named_modules()))
        assert [(node.name, sites) for node, sites in shared] == [
            ("wx", ["wx", "wx_1", "wx_2"]), ("wh", ["wh", "wh_1"]),
        ]

    def test_no_shared_state_without_reuse(self):
        gm = trace_model(_ParamPlusActivation().eval(), (4,))
        assert shared_state_sites(gm, dict(gm.named_modules())) == []


# ----------------------------------------------------------------------------- constants are not ties
class ConstantShift(nn.Module):
    """A buffer-only module: a constant shift (no trainable state to tie)."""

    def __init__(self, c: torch.Tensor):
        super().__init__()
        self.register_buffer("c", c)

    def forward(self, x):
        return x + self.c


class ReusedConstantShift(nn.Module):
    """One buffer-only module applied at two call sites (``tied=False``: two equal buffers, one per site)."""

    def __init__(self, tied: bool = True):
        super().__init__()
        torch.manual_seed(3)
        c = torch.randn(6)
        self.l1, self.l2, self.head = nn.Linear(6, 6), nn.Linear(6, 6), nn.Linear(6, 3)
        self.shift = ConstantShift(c)
        self.other = self.shift if tied else ConstantShift(c.clone())

    def forward(self, x):
        return self.head(self.l2(torch.relu(self.shift(self.l1(x)))) + self.other(x))


class ReusedAffineFreeBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.head = nn.Linear(6, 6), nn.Linear(6, 6), nn.Linear(6, 3)
        self.bn = nn.BatchNorm1d(6, affine=False)

    def forward(self, x):
        return self.head(torch.relu(self.bn(self.l1(x))) + self.bn(self.l2(x)))


def _randomized_stats(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 2.0)
    return model


@pytest.mark.parametrize("packaging_name", sorted(PACKAGINGS))
@pytest.mark.parametrize("factory", [ReusedConstantShift, ReusedAffineFreeBatchNorm], ids=["shift", "bn_affine_false"])
def test_a_reused_buffer_only_module_converts_exactly(factory, packaging_name):
    torch.manual_seed(0)
    model = _randomized_stats(factory().double().eval())
    report = check_representability(model, (6,), packaging=PACKAGINGS[packaging_name])
    assert report.is_representable and not _sharing_ops(report)
    flow = convert_torch_model(model, (6,), 3, packaging=PACKAGINGS[packaging_name]).eval()
    x = torch.randn(4, 6, dtype=torch.float64)
    with torch.no_grad():
        assert float((flow(x) - model(x)).abs().max()) <= 1e-12


def _canonical_state(flow) -> list:
    return [(key, str(value.dtype), tuple(value.shape), value.detach().cpu().numpy().tobytes())
            for key, value in flow.state_dict().items()]


def _ir_digest(flow) -> list:
    from mimarsinan.mapping.ir import ComputeOp, NeuralCore
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    ir = IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
                   allow_coalescing=False, hardware_bias=False).map(flow.get_mapper_repr())
    rows = []
    for node in ir.nodes:
        if isinstance(node, NeuralCore):
            rows.append(("core", node.name, node.get_core_matrix(ir).tobytes()))
        elif isinstance(node, ComputeOp):
            rows.append(("op", node.name, node.op_type, tuple(node.input_shape or ()), tuple(node.output_shape or ())))
    return rows


@pytest.mark.parametrize("packaging_name", sorted(PACKAGINGS))
def test_a_reused_constant_converts_as_its_untied_twin(packaging_name):
    """Sharing a constant buffer cannot change the conversion: parameters and IR equal the untied program's."""
    packaging = PACKAGINGS[packaging_name]
    flows = [convert_torch_model(ReusedConstantShift(tied).double().eval(), (6,), 3, packaging=packaging)
             for tied in (True, False)]
    assert _canonical_state(flows[0]) == _canonical_state(flows[1])
    assert _ir_digest(flows[0]) == _ir_digest(flows[1])
