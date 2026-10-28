"""Lane roles keyed by call site (U-K2; U-K review m7): one module object called at several sites of a network (a
block calling its ReLU after each of its two sums, as a residual block may) gives each site its own role, so a site
the lane plan absorbs fuses and a site that reads a host value stays on the host. Each program is compared at atol 0
with the function-identical twin that holds one module per site, and certified NF = SCM = HCM = int64 golden."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource
from mimarsinan.mapping.lanes.install import LANE_REPORT
from mimarsinan.mapping.lanes.ir.refusal import LaneFusionError
from unit.chip_simulation.value_run.integer_programs import _randomize_bn, assert_exact
from unit.mapping.lanes.lane_programs import build_lane_program, host_ops, lane_platform, real_inputs


class _Block(nn.Module):
    """A basic block whose two activations are one ReLU module called twice (``shared``) or one module per call
    (the twin); a projection shortcut where the shape changes."""

    def __init__(self, cin: int, cout: int, stride: int, shared: bool) -> None:
        super().__init__()
        self.shared = shared
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU()
        if not shared:
            self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return (self.relu if self.shared else self.relu_out)(out + self.shortcut(x))


class _SharedActivationNet(nn.Module):
    """A stem ending in a max pool (a host op under every lane class: the first block's sum reads a host operand, so
    it and the second call of that block's ReLU stay on the host), then a projection block whose sum fuses under E
    (both calls of its ReLU are lanes then), an average pool and a linear head."""

    def __init__(self, shared: bool) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool0 = nn.MaxPool2d(2)
        self.block1 = _Block(8, 8, 1, shared)
        self.block2 = _Block(8, 16, 2, shared)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.block2(self.block1(self.pool0(self.relu(self.bn(self.stem(x))))))
        return self.fc(torch.flatten(self.pool(x), 1))


SHAPE = (3, 16, 16)
#: class set -> (host op kinds left on the host, the host ReLU call sites of the shared network, the fusion report's
#: (clamps, adds, pools)); the twin (one module per site) gives the same kinds and counts on RC0.
EXPECTED = {
    ("L",): (["MaxPool2d", "_operator.add", "ReLU", "_operator.add", "ReLU", "AdaptiveAvgPool2d"],
             ["block1_relu_1", "block2_relu_1"], (3, 0, 0)),
    ("L", "E"): (["MaxPool2d", "_operator.add", "ReLU", "AdaptiveAvgPool2d"], ["block1_relu_1"], (4, 1, 0)),
    ("L", "E", "P"): (["MaxPool2d", "_operator.add", "ReLU"], ["block1_relu_1"], (4, 1, 1)),
}
GEOMETRIES = (dict(axons=256, neurons=32), dict(axons=256, neurons=32, count=24, passes=128))


def _net(shared: bool) -> nn.Module:
    torch.manual_seed(5)
    return _randomize_bn(_SharedActivationNet(shared))


def _program(shared: bool, classes):
    return build_lane_program(_net(shared), SHAPE, num_classes=10, platform=lane_platform(classes))


def _run(flow, x):
    with torch.no_grad():
        return flow(x)


def _host_relus(program) -> list:
    return [n.name for n in program.ir.nodes if isinstance(n, ComputeOp) and n.op_type == "ReLU"]


@pytest.fixture(scope="module", params=sorted(EXPECTED), ids=lambda c: ",".join(c))
def pair(request):
    return request.param, _program(True, request.param), _program(False, request.param)


class TestSharedActivationModule:
    def test_each_call_site_of_one_module_takes_its_own_role(self, pair):
        classes, shared, twin = pair
        kinds, relus, counts = EXPECTED[classes]
        report = getattr(shared.ir, LANE_REPORT)
        assert host_ops(shared) == kinds == host_ops(twin)
        assert _host_relus(shared) == relus, "only the call sites the plan leaves on the host stay there"
        assert (report["clamps"], report["adds"], report["pools"]) == counts
        twin_report = getattr(twin.ir, LANE_REPORT)
        assert {k: report[k] for k in ("clamps", "adds", "pools", "reductions", "operand_rows")} == \
            {k: twin_report[k] for k in ("clamps", "adds", "pools", "reductions", "operand_rows")}

    def test_the_program_equals_its_twin_and_certifies_at_atol_zero(self, pair):
        _classes, shared, twin = pair
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

        x = real_inputs(shared, 32)
        scm = _run(shared.identity(), x)
        assert torch.equal(scm, _run(twin.identity(), x)), "one module per site is the same function"
        assert_exact(_run(shared.nf(), x), scm)
        for geometry in GEOMETRIES:
            mapping = shared.packed_mapping(**geometry)
            hcm = _run(ValueHybridCoreFlow(mapping, dtype=torch.float64), x)
            assert_exact(hcm, scm)
            codes, steps = run_golden(mapping, x.numpy())
            assert np.array_equal(codes * steps, hcm.reshape(codes.shape).numpy())


def test_the_off_ablation_of_a_shared_module_is_the_lane_free_program():
    """Control (green before the change too): with the lanes off nothing is fused, so the IR equals the lane-free
    program's and so do its outputs."""
    from unit.mapping.value_grid_identity import _ir_digest

    plain, off = _program(True, None), _program(True, [])
    assert _ir_digest(off.ir) == _ir_digest(plain.ir)
    x = real_inputs(plain, 8)
    assert torch.equal(_run(off.identity(), x), _run(plain.identity(), x))


def _op(node_id: int, name: str, module) -> ComputeOp:
    return ComputeOp(id=node_id, name=name, input_sources=np.array([IRSource(0, 0)], dtype=object),
                     op_type=type(module).__name__, params={"module": module})


def _ir(*ops) -> IRGraph:
    return IRGraph(nodes=list(ops), output_sources=np.array([IRSource(int(ops[-1].id), 0)], dtype=object))


class TestCallSiteRoles:
    """The resolver alone: a site's op carries the site's payload under the site's name."""

    def test_a_named_site_takes_the_op_of_its_name_only(self):
        from mimarsinan.mapping.lanes.ir.call_sites import call_site_roles

        relu = nn.ReLU()
        ir = _ir(_op(1, "relu", relu), _op(2, "relu_1", relu))
        roles, carried = call_site_roles(ir, [(SimpleNamespace(module=relu, name="relu_1"), ("clamp", None))])
        assert roles == {2: ("clamp", None)} and carried == 1

    def test_two_sites_of_one_module_take_one_op_each(self):
        from mimarsinan.mapping.lanes.ir.call_sites import call_site_roles

        relu = nn.ReLU()
        ir = _ir(_op(1, "relu", relu), _op(2, "relu_1", relu))
        sites = [(SimpleNamespace(module=relu, name=n), ("clamp", None)) for n in ("relu", "relu_1")]
        assert call_site_roles(ir, sites) == ({1: ("clamp", None), 2: ("clamp", None)}, 2)

    def test_an_unnamed_site_is_told_by_its_module_when_one_op_carries_it(self):
        from mimarsinan.mapping.lanes.ir.call_sites import call_site_roles

        relu, pool = nn.ReLU(), nn.AvgPool2d(2)
        ir = _ir(_op(1, "relu", relu), _op(2, "compute_AvgPool2d_2", pool))
        assert call_site_roles(ir, [(SimpleNamespace(module=pool, name=None), ("pool", 7))]) == ({2: ("pool", 7)}, 1)

    def test_an_unnamed_site_whose_module_runs_at_two_ops_is_refused_by_name(self):
        from mimarsinan.mapping.lanes.ir.call_sites import call_site_roles

        relu = nn.ReLU()
        ir = _ir(_op(1, "compute_ReLU_1", relu), _op(2, "compute_ReLU_2", relu))
        with pytest.raises(LaneFusionError, match="no name that tells them apart"):
            call_site_roles(ir, [(SimpleNamespace(module=relu, name=None), ("clamp", None))])

    def test_a_site_the_ir_does_not_carry_is_not_counted(self):
        from mimarsinan.mapping.lanes.ir.call_sites import call_site_roles

        relu = nn.ReLU()
        ir = _ir(_op(1, "relu", relu))
        assert call_site_roles(ir, [(SimpleNamespace(module=relu, name="relu_7"), ("clamp", None))]) == ({}, 0)
