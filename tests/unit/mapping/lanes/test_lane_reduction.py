"""WI-5e producer-side reduction (D11): an average pool read alone from a perceptron's open lane reduces into that
perceptron's accumulator, one rounding for the pooled value (arith_toggle.md section 3.4), certified at atol 0."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.lanes.declaration import parse_lane_declaration
from mimarsinan.mapping.lanes.install import LANE_REPORT
from mimarsinan.mapping.lanes.ir.reduce_cores import REDUCED_INSTANCES
from mimarsinan.mapping.lanes.ir.refusal import LaneFusionError
from mimarsinan.mapping.lanes.plan import plan_lanes
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from unit.chip_simulation.value_run.integer_programs import _randomize_bn, assert_exact
from unit.mapping.lanes.lane_programs import build_lane_program, cores, host_ops, lane_platform, real_inputs


class _Transition(nn.Module):
    """A DenseNet-BC transition shape: conv-BN-ReLU; a 1x1 conv-BN read only by a 2x2 average pool (reduced into
    it); a biased 3x3 conv read only by an overlapping, zero-padded 3x3 / 2 pool and then a ReLU (reduced, then
    clamped); a global pool over that ReLU's lane (a pool lane of its own) and a linear head."""

    def __init__(self, shared: bool = False) -> None:
        super().__init__()
        self.shared = shared
        self.conv0 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(8)
        self.relu0 = nn.ReLU()
        self.trans = nn.Conv2d(8, 6, 1, bias=False)
        self.bn_t = nn.BatchNorm2d(6)
        self.pool_t = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)
        self.pool2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 + (6 if shared else 0), 10)

    def forward(self, x):
        x = self.relu0(self.bn0(self.conv0(x)))
        t = self.bn_t(self.trans(x))
        y = self.relu2(self.pool2(self.conv2(self.pool_t(t))))
        head = torch.flatten(self.gap(y), 1)
        if self.shared:  # the transition's output read beside its pool: no reduction into it
            head = torch.cat([head, torch.flatten(self.gap(torch.relu(t)), 1)], dim=1)
        return self.fc(head)


def transition_net(shared: bool = False) -> nn.Module:
    torch.manual_seed(21)
    return _randomize_bn(_Transition(shared))


SHAPE = (3, 16, 16)


def _program(**kwargs):
    return build_lane_program(transition_net(), SHAPE, num_classes=10, platform=lane_platform(), **kwargs)


def _run(flow, x):
    with torch.no_grad():
        return flow(x)


class TestThePlan:
    def test_a_pool_read_alone_from_an_open_lane_is_its_reduction(self):
        flow = convert_torch_model(transition_net(), SHAPE, 10, device="cpu", packaging=MVM_PACKAGING).eval()
        plan = plan_lanes(flow.get_mapper_repr(), parse_lane_declaration({"classes": ["L", "E", "P"]}))
        reduced = [lane for lane in plan.perceptron_lanes if lane.reduce is not None]
        assert len(reduced) == 2 and len(plan.pool_lanes) == 1
        assert [lane.clamp is not None for lane in reduced] == [False, True], "the second reduction is clamped"
        assert plan.counts()["reduce"] == 2

    def test_a_producer_read_beside_its_pool_keeps_its_rounding(self):
        flow = convert_torch_model(transition_net(shared=True), SHAPE, 10, device="cpu",
                                   packaging=MVM_PACKAGING).eval()
        plan = plan_lanes(flow.get_mapper_repr(), parse_lane_declaration({"classes": ["L", "E", "P"]}))
        assert sum(lane.reduce is not None for lane in plan.perceptron_lanes) == 1  # only conv2's pool

    def test_without_p_no_pool_reduces(self):
        flow = convert_torch_model(transition_net(), SHAPE, 10, device="cpu", packaging=MVM_PACKAGING).eval()
        plan = plan_lanes(flow.get_mapper_repr(), parse_lane_declaration({"classes": ["L", "E"]}))
        assert not any(lane.reduce is not None for lane in plan.perceptron_lanes) and not plan.pool_lanes


class TestTheReducedCores:
    def test_every_pool_runs_on_chip_and_the_reductions_replace_their_producers(self):
        program = _program()
        assert host_ops(program) == []
        report = getattr(program.ir, LANE_REPORT)
        assert (report["reductions"], report["pools"]) == (2, 1)
        reduced = [c for c in cores(program) if getattr(c, REDUCED_INSTANCES, None) is not None]
        assert reduced and {int(c.perceptron_index) for c in reduced} == {1, 2}
        assert not [c for c in cores(program) if c.perceptron_index == 1 and c not in reduced], \
            "the 1x1 producer's own cores left the program"

    def test_a_reduced_core_stacks_its_window_instances_on_the_producer_weights(self):
        program = _program()
        blocks_seen = 0
        for core in cores(program):
            instances = getattr(core, REDUCED_INSTANCES, None)
            if instances is None:
                continue
            matrix = np.asarray(core.get_core_matrix(program.ir))
            n = int(instances[0])
            assert np.all(instances == n) and matrix.shape[0] % n == 0, "one equal row block per window instance"
            assert int(np.abs(matrix).max()) <= 7, "4-bit weights: stacking keeps every code on the producer's grid"
            if core.perceptron_index == 1:  # the 1x1 conv under 2x2 windows: four equal blocks of its codes
                blocks = matrix.reshape(n, matrix.shape[0] // n, matrix.shape[1])
                assert n == 4 and all(np.array_equal(blocks[0], b) for b in blocks)
                blocks_seen += 1
        assert blocks_seen == 64, "one reduced core per 2x2 window of the 16 x 16 map"

    def test_theta_carries_the_window_size(self):
        program = _program()
        from mimarsinan.mapping.lanes.ir.reduce_cores import PRODUCER_THETA

        reduced = [c for c in cores(program) if getattr(c, REDUCED_INSTANCES, None) is not None]
        divisor = {1: 4.0, 2: 9.0}  # 2x2, and 3x3 counting zero padding
        for core in reduced:
            producer = float(getattr(core, PRODUCER_THETA))
            assert float(core.threshold) == divisor[int(core.perceptron_index)] * producer
        perceptrons = list(program.model.get_perceptrons())
        assert {float(getattr(c, PRODUCER_THETA)) for c in reduced if c.perceptron_index == 1} == {
            float(perceptrons[1].parameter_scale)}


@pytest.mark.parametrize("geometry", [
    dict(axons=512, neurons=64),
    dict(axons=512, neurons=32, count=24, passes=256),
    dict(axons=128, neurons=32, coalescing=True),
])
@pytest.mark.parametrize("hardware_bias", [False, True])
def test_nf_equals_scm_equals_hcm_equals_the_golden_at_atol_zero(geometry, hardware_bias):
    program = _program(hardware_bias=hardware_bias)
    x = real_inputs(program, 24)
    scm = _run(program.identity(), x)
    assert_exact(_run(program.nf(), x), scm)
    mapping = program.packed_mapping(**geometry)
    if geometry.get("passes"):
        assert sum(stage.kind == "neural" for stage in mapping.stages) > 1, "the declared cores run several passes"
    assert_exact(_run(ValueHybridCoreFlow(mapping, dtype=torch.float64, device="cpu"), x), scm)
    codes, steps = run_golden(mapping, x.numpy())
    assert np.array_equal(codes * steps, scm.reshape(codes.shape).numpy())


def test_the_reduction_rounds_once():
    """The reduced lane's codes are one rounding of the pooled accumulator: the requantize-first form (round each
    instance onto the lane grid, then pool) gives other codes on the same weights."""
    program = _program()
    model = program.nf()
    perceptrons = list(model.get_perceptrons())
    seen = {}

    def keep(name):
        def hook(module, args, out):
            seen[name] = out.detach()
        return hook

    flow_modules = dict(model.named_modules())
    pool = next(m for m in flow_modules.values() if type(m).__name__ == "LaneReduceModule")
    handles = [pool.register_forward_hook(keep("reduced"))]
    try:
        _run(model, real_inputs(program, 16))
    finally:
        for handle in handles:
            handle.remove()
    step = float(perceptrons[1].scaler.range) / 7.0
    codes = seen["reduced"] / step
    assert float((codes - torch.round(codes)).abs().max()) < 1e-9, "the reduced value is a code of the lane grid"


def test_a_pruned_producer_refuses_by_name():
    with pytest.raises(LaneFusionError, match="pruned"):
        build_lane_program(transition_net(), SHAPE, num_classes=10, platform=lane_platform(), prune={1: [0, 2]})


def test_the_off_ablation_and_the_float_path_keep_every_pool_on_the_host():
    off = build_lane_program(transition_net(), SHAPE, num_classes=10, platform=lane_platform([]))
    assert host_ops(off).count("AvgPool2d") == 2 and "AdaptiveAvgPool2d" in host_ops(off)
    assert not any(isinstance(n, NeuralCore) and getattr(n, REDUCED_INSTANCES, None) is not None
                   for n in off.ir.nodes)
