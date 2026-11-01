"""[U-AX, SDC-D1] Plane parity on a per-axon pool: the exploration plane packs the physical rows the build packs.

Phase C's SD-11 on the release head read densenet_bc100 on ODIN as disagreeing on payload bytes per period and write
energy: per placed softcore the candidate counted one row more, the always-on row of a bias-free convolution, all zero.
The layout walk planned two rows for every input under per-axon sign, while the build's sign split takes one row for an
all-zero or single-sign row and two for a mixed one (1 + mixed). Where the walk holds the network's weights, it reads
each core's rows by that rule on the grid the build reads, so the two planes count the same rows.

The vehicles are small spiking programs on ODIN at a scheduled budget (an eighth of F, as the harness samples):
(a) bias-free convolutions, each instance's always-on row all zero; (b) convolutions whose rows of two input channels
hold one sign; (e) rows mixed in float whose small negative weights round to zero on the declared 4-bit grid; (f) a
network trained on its integer lattice, built at its trained scales; the guard (c) is ResNet-18 w1/4 (CL5's
resnet18_w1q, under the harness's coalescing departure), every row mixed, whose rows are unchanged.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from unit.chip_simulation.value_run.integer_programs import resnet18_quarter

from . import per_axon_planes as PA

SHAPE = (3, 4, 4)
#: The harness's labelled departure for a bundle that declares no coalescing (CL5's ``+coalescing`` jobs)
DEPARTURE = {"allow_coalescing": True}


class _Convs(nn.Module):
    """An encoding convolution (on the host under the spiking contract), two on-chip 3 x 3 convolutions and a head."""

    def __init__(self, bias: bool):
        super().__init__()
        self.c0 = nn.Conv2d(3, 8, 3, padding=1)
        self.c1 = nn.Conv2d(8, 8, 3, padding=1, bias=bias)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1, bias=bias)
        self.relu, self.flat, self.fc = nn.ReLU(), nn.Flatten(), nn.Linear(8 * 16, 10)

    def forward(self, x):
        x = self.relu(self.c0(x))
        return self.fc(self.flat(self.relu(self.c2(self.relu(self.c1(x))))))


def _vehicle(kind: str) -> nn.Module:
    torch.manual_seed(0)
    model = _Convs(bias=kind not in ("bias_free", "trained"))
    with torch.no_grad():
        if kind == "single_sign":  # input channels 0 and 1: every weight onto every output positive
            for conv in (model.c1, model.c2):
                conv.weight[:, :2] = conv.weight[:, :2].abs()
        if kind == "grid_zeroed":  # input channel 0: positive but one weight per row, -1e-4, which the grid rounds to 0
            for conv in (model.c1, model.c2):
                conv.weight[:, 0] = conv.weight[:, 0].abs()
                conv.weight[0, 0] = -1e-4
    return model


def _trained(net):
    """Every on-chip perceptron on its integer lattice at an integer scale (NAPQ's single-scale install, by hand)."""
    for perceptron in net.mapper_repr.get_perceptrons():
        with torch.no_grad():
            weight, bias = perceptron.layer.weight, perceptron.layer.bias
            peak = float(weight.abs().max()) if bias is None else max(float(weight.abs().max()),
                                                                     float(bias.abs().max()))
            scale = max(1.0, math.floor(7.0 / peak))
            weight.copy_(torch.round(weight * scale) / scale)
            if bias is not None:
                bias.copy_(torch.round(bias * scale) / scale)
        perceptron.set_parameter_scale(torch.tensor(scale))
    return net


def _pair(kind: str, **kwargs):
    net = PA.network(_vehicle(kind), SHAPE, 10, kind)
    if kind == "trained":
        net = _trained(net)
    return PA.pair(net, trained=kind == "trained", **kwargs)


def _assert_planes_agree(p: PA.Pair) -> None:
    assert p.candidate_sign_rows == p.executed_sign_rows, (sum(p.candidate_sign_rows), sum(p.executed_sign_rows))
    assert p.candidate_cells == p.executed_cells
    assert p.candidate["payload_bytes_per_period"] == p.executed["payload_bytes_per_period"]
    assert p.candidate["energy_write_mj"] == pytest.approx(p.executed["energy_write_mj"], rel=1e-9)
    assert p.planes["agree"], [f for f in p.planes["fields"] if not f["agree"]]


@pytest.mark.parametrize("kind", ["bias_free", "single_sign", "grid_zeroed", "trained"])
def test_the_candidate_states_the_executed_programs_rows_payload_and_write_energy(kind):
    """(a), (b), (e), (f): a scheduled point (written passes, so the payload and the write energy are priced); the
    candidate's sign rows, cells, payload bytes per period and write energy are the executed program's."""
    p = _pair(kind)
    assert p.executed["written_passes"] and p.executed["payload_bytes_per_period"], p.executed
    assert p.executed["energy_write_mj"] is not None
    _assert_planes_agree(p)
    assert p.point["physical_rows"] == {"basis": "deployed", "reason": None}


@pytest.mark.slow
def test_resnet18_quarter_on_odin_keeps_two_rows_per_input_and_agrees():
    """(c) Guard: every row of ResNet-18 w1/4 is mixed (SD-11: no non-mixed row), so its rows are two per input, as
    the weight-free walk plans them, and the planes agree at the harness's sampled point."""
    torch.manual_seed(0)
    net = PA.network(resnet18_quarter(), (3, 32, 32), 10, "resnet18_w1q")
    p = PA.pair(net, declared=DEPARTURE)
    chip = PA.script().resolved_platform(PA.TARGET, "declared", None, dict(DEPARTURE), {})
    pm = resolve_platform_mapping_params(chip["cores"], allow_coalescing=True)
    logical = LayoutIRMapping(max_axons=pm.effective_max_axons, max_neurons=pm.effective_max_neurons,
                              allow_coalescing=True, hardware_bias=pm.hardware_bias).collect_layout_softcores(
        net.mapper_repr)
    assert [int(s.input_count) for s in p.census.softcores] == [2 * int(s.input_count) for s in logical]
    _assert_planes_agree(p)
    assert p.point.get("physical_rows", {"basis": "deployed"})["basis"] == "deployed"
