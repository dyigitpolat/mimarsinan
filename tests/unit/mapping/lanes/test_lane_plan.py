"""WI-5e: the lane plan on the mapper graph (placement by declared class, one rounding per lane) and the
segment entries it leaves to Boundary Quantization."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.lanes.declaration import parse_lane_declaration
from mimarsinan.mapping.lanes.model_lanes import install_model_lanes
from mimarsinan.mapping.lanes.plan import plan_lanes
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.models.nn.activations.lane_quantization import LaneAddModule, LanePoolModule
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from unit.chip_simulation.value_run.integer_programs import lenet5, mlp, residual_block_net


def _flow(model, shape, classes=10):
    return convert_torch_model(model.eval(), shape, classes, device="cpu", packaging=MVM_PACKAGING).eval()


def _plan(flow, classes="LEP"):
    return plan_lanes(flow.get_mapper_repr(), parse_lane_declaration({"classes": list(classes)}))


class _SharedPreAdd(nn.Module):
    """``a``'s output feeds the add and a second reader, so the add accumulates into ``b`` with ``a`` as operand
    (behind an on-chip stem; ``host_fed``: ``a`` and ``b`` read the network input, a host value)."""

    def __init__(self, host_fed: bool = False):
        super().__init__()
        self.host_fed = host_fed
        self.stem = nn.Conv2d(2, 2, 1)
        self.a = nn.Conv2d(2, 2, 3, padding=1)
        self.b = nn.Conv2d(2, 2, 3, padding=1)
        self.c = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        x = x if self.host_fed else torch.relu(self.stem(x))
        y = self.a(x)
        return torch.relu(y + self.b(x)) + self.c(y)


class _SharedOperands(nn.Module):
    """Both inputs of the first add have second readers: no core can hold the sum, so it stays on the host (and so
    do the adds that read it, whose operand is then a host value)."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Conv2d(2, 2, 3, padding=1), nn.Conv2d(2, 2, 3, padding=1)
        self.c, self.d = nn.Conv2d(2, 2, 1), nn.Conv2d(2, 2, 1)

    def forward(self, x):
        y, z = self.a(x), self.b(x)
        return (y + z) + self.c(y) + self.d(z)


class TestPlan:
    def test_the_residual_network(self):
        plan = _plan(_flow(residual_block_net(), (3, 16, 16)))
        assert plan.counts() == {"clamp": 5, "add": 2, "pool": 1, "reduce": 0}
        adds = [lane for lane in plan.perceptron_lanes if lane.add is not None]
        assert all(lane.clamp is not None and lane.unsigned for lane in adds), "each add lane ends at its ReLU"
        assert [lane.operand_index for lane in adds] == [1, 1], "the add's second input is the operand"
        assert plan.pool_lanes[0].unsigned, "a pool over a clamped lane is unsigned"

    def test_classes_gate_their_ops(self):
        flow = _flow(residual_block_net(), (3, 16, 16))
        assert _plan(flow, "L").counts() == {"clamp": 3, "add": 0, "pool": 0, "reduce": 0}
        assert _plan(flow, "LE").counts() == {"clamp": 5, "add": 2, "pool": 0, "reduce": 0}

    def test_max_pools_stay_on_the_host(self):
        plan = _plan(_flow(lenet5(), (1, 28, 28)))
        assert plan.counts() == {"clamp": 4, "add": 0, "pool": 0, "reduce": 0}

    def test_an_add_accumulates_into_the_producer_it_alone_reads(self):
        plan = _plan(_flow(_SharedPreAdd(), (2, 6, 6), classes=2))
        adds = [lane for lane in plan.perceptron_lanes if lane.add is not None]
        assert [lane.holder.name for lane in adds] == ["b", "c"] and [lane.operand_index for lane in adds] == [0, 0]
        assert adds[0].clamp is not None and adds[1].clamp is None

    def test_an_add_into_a_perceptron_reading_a_host_value_stays_on_the_host(self):
        """M1: ``b`` reads the network input on its entry grid, which an operand row read on ``a``'s grid cannot
        share, so the add stays on the host, and with it the add that reads its ReLU."""
        plan = _plan(_flow(_SharedPreAdd(host_fed=True), (2, 6, 6), classes=2))
        assert plan.counts()["add"] == 0

    def test_an_add_with_no_exclusive_producer_stays_on_the_host(self):
        plan = _plan(_flow(_SharedOperands(), (2, 6, 6), classes=2))
        assert plan.counts()["add"] == 0

    def test_the_plan_survives_its_installation(self):
        flow = _flow(residual_block_net(), (3, 16, 16))
        install_readout_quantizers(list(flow.get_perceptrons()), 4)
        plan = _plan(flow)
        install_model_lanes(plan, 4)
        modules = [type(node.module) for node in flow.get_mapper_repr().execution_order()
                   if hasattr(node, "module")]
        assert modules.count(LaneAddModule) == 2 and modules.count(LanePoolModule) == 1
        assert _plan(flow).counts() == plan.counts(), "the installed lane ends are recognized again"


class TestEntries:
    def test_a_perceptron_reading_a_lane_is_no_entry(self):
        flow = _flow(residual_block_net(), (3, 16, 16))
        plain = segment_entry_perceptrons(flow.get_mapper_repr())
        lanes = segment_entry_perceptrons(flow.get_mapper_repr(), _plan(flow).absorbed())
        assert len(plain) == 7 and len(lanes) == 1, "only the stem reads the host"

    def test_without_lanes_the_entries_are_unchanged(self):
        flow = _flow(mlp(), (8,), classes=4)
        assert segment_entry_perceptrons(flow.get_mapper_repr(), frozenset()) == segment_entry_perceptrons(
            flow.get_mapper_repr())
