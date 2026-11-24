"""The residual sum rounded once (branch tcad/residual-sum-rounding, PL/rsum/design.md).

Under executed lanes a constant-zero pad over an on-chip value is on-chip wiring: the plan counts it on chip and the
fusion replaces it by its input's sources with an off source at each padded element (an off operand adds 0). So a
residual add whose shortcut is zero-padded in channels takes the on-chip addition the lane plan already gives an add
with an on-chip operand, and its sum is rounded once on the sum's grid, as the training code rounds it. An add that
the declared class E would put on chip but that stays on the host (its operand another host value, or its holder a
segment entry) while the declared state gives its holder a readout range is refused by name: that range would round
the branch, never the sum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.lanes.install import LANE_REPORT
from mimarsinan.models.families.pyramidnet import CifarPyramidNet
from unit.chip_simulation.value_run.integer_programs import assert_exact
from unit.mapping.lanes.lane_programs import build_lane_program, host_ops, lane_platform, real_inputs, write_state

ADD_OP = "_operator.add"


def _four_values(program, x: torch.Tensor, **packed):
    """NF, SCM, HCM and the int64 golden on ``x`` (fp64), in that order (the golden runs last: it reads only a
    program whose outputs are cores)."""
    with torch.no_grad():
        yield program.nf()(x)
        yield program.identity()(x)
        mapping = program.packed_mapping(**packed)
        yield ValueHybridCoreFlow(mapping, dtype=torch.float64, device="cpu")(x)
    codes, steps = run_golden(mapping, x.numpy())
    yield torch.as_tensor(codes * steps)


def _plan(program) -> dict:
    return dict(getattr(program.ir, LANE_REPORT)["plan"])


# ------------------------------------------------------------------------------------------- 1. the hand example
class _PaddedSum(nn.Module):
    """A stem ``s = relu(x)`` (2 channels), a branch ``b = W s + c`` (3 channels, no activation after the sum) and
    the shortcut ``s`` zero-padded by one channel: ``b + pad(s)``. Hand weights: ``b0 = 5 s1 - 0.4``, ``b1 = s0``,
    ``b2 = s0 + s1``."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(2, 2, 1)
        self.branch = nn.Conv2d(2, 3, 1)
        with torch.no_grad():
            self.stem.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            self.stem.bias.zero_()
            self.branch.weight.copy_(torch.tensor([[0.0, 5.0], [1.0, 0.0], [1.0, 1.0]]).reshape(3, 2, 1, 1))
            self.branch.bias.copy_(torch.tensor([-0.4, 0.0, 0.0]))

    def forward(self, x):
        s = torch.relu(self.stem(x))
        return (self.branch(s) + F.pad(s, (0, 0, 0, 0, 0, 1))).flatten(1)


def _hand_program(tmp_path):
    """The stem's entry and readout on step 0.2 (codes 0..15), the sum's grid declared on the branch: step 1, offset
    0, codes 0..15."""
    platform = lane_platform(("L", "E", "P"), activation_bits=4)

    def declare(flow, entries, grids, perceptrons, quantizers, lanes):
        platform["quantizer_state"] = write_state(Path(tmp_path) / "state.json", {
            "stem": {"entry": {"scale": 3.0, "codes": [0, 15]}, "readout_range": 3.0},
            "branch": {"readout_range": 15.0, "readout_offset": 0.0}})

    return build_lane_program(_PaddedSum(), (2, 1, 1), num_classes=3, platform=platform, weight_bits=8,
                              state_hook=declare)


#: (s0, s1) per input: branch 0.6 with shortcut 0.6, and branch 14.6 with shortcut 3 (the clip).
HAND_INPUTS = torch.tensor([[0.6, 0.2], [3.0, 3.0]], dtype=torch.float64).reshape(2, 2, 1, 1)
#: The sum rounded once on step 1, codes 0..15 (the training code): 1.2 -> 1, 0.8 -> 1, 17.6 -> 15, 6 -> 6. The paper
#: version rounds the branch and then adds the shortcut: [[1.6, 1.2, 1], [18, 6, 6]].
ROUNDED_ONCE = torch.tensor([[1.0, 1.0, 1.0], [15.0, 6.0, 6.0]], dtype=torch.float64)


class TestTheHandExample:
    def test_nf_scm_hcm_and_the_golden_round_the_sum_once(self, tmp_path):
        program = _hand_program(tmp_path)
        for name, value in zip(("NF", "SCM", "HCM", "golden"),
                               _four_values(program, HAND_INPUTS, axons=64, neurons=8)):
            assert torch.equal(value.reshape(ROUNDED_ONCE.shape), ROUNDED_ONCE), (name, value.reshape(2, 3))

    def test_the_pad_and_the_add_leave_the_host(self, tmp_path):
        program = _hand_program(tmp_path)
        assert host_ops(program) == []
        assert _plan(program)["add"] == 1 and _plan(program)["pad"] == 1


# ------------------------------------------------------------------------------------------- 2. PyramidNet
def _pyramid_program(alpha: int, platform=None, **mapping):
    """PyramidNet depth 8 on 3 x 16 x 16 with non-trivial BN statistics (the design's probe, PL/rsum/design)."""
    torch.manual_seed(0)
    model = CifarPyramidNet((3, 16, 16), 10, depth=8, alpha=alpha)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
    return build_lane_program(model, (3, 16, 16), num_classes=10, platform=platform or lane_platform(), **mapping)


#: What alpha 0 (no pad) leaves on the host: the three units' input BN, the final BN and ReLU, the head pool.
ALPHA0_HOST = ["BatchNormAffine"] * 4 + ["ReLU", "AdaptiveAvgPool2d"]


@pytest.fixture(scope="module", params=[84, 0], ids=["alpha84_padded", "alpha0_unpadded"])
def pyramid(request):
    return request.param, _pyramid_program(request.param)


class TestPyramidNet:
    def test_every_add_takes_its_on_chip_addition_and_each_pooled_shortcut_its_pool_lane(self, pyramid):
        alpha, program = pyramid
        plan = _plan(program)
        assert (plan["add"], plan["pool"], plan.get("pad")) == (3, 2, 3 if alpha else None), plan
        assert host_ops(program) == ALPHA0_HOST

    def test_nf_equals_scm_equals_hcm_equals_the_golden_at_atol_zero(self, pyramid):
        _alpha, program = pyramid
        nf, scm, hcm, golden = _four_values(program, real_inputs(program, 16), axons=1024, neurons=256)
        assert_exact(nf, scm)
        assert_exact(hcm, scm)
        assert np.array_equal(golden.numpy(), hcm.reshape(golden.shape).numpy())


@pytest.mark.parametrize("geometry", [dict(axons=1024, neurons=64, count=4096),
                                      dict(axons=1024, neurons=32, count=8, passes=512)], ids=["one_pool", "scheduled"])
def test_the_shape_only_plane_walks_the_pads_as_the_build_fuses_them(geometry):
    """The exploration's shape-only plane (``shape_walk.py``) applies the pads as wiring, as the fusion does: both
    planes state one lane program (``test_lane_shape_walk``'s facts), with alpha 0's six host ops."""
    from unit.mapping.lanes.lane_planes import builder_plane, candidate_plane, ir_geometry, plane_platform

    platform = plane_platform(("L", "E", "P"), **geometry)
    ir = ir_geometry(platform)
    program = _pyramid_program(84, lane_platform(cores=[dict(core) for core in platform["cores"]]),
                               max_axons=ir["max_axons"], max_neurons=ir["max_neurons"],
                               hardware_bias=ir["hardware_bias"], allow_coalescing=ir["allow_coalescing"])
    candidate, builder = candidate_plane(program, platform, 10), builder_plane(program, platform)
    for key in ("segments", "host_ops", "specs", "passes", "tiles", "cells", "written", "carry"):
        assert candidate[key] == builder[key], f"{key}: shape-only plane {candidate[key]} != builder {builder[key]}"
    assert builder["host_ops"] == len(ALPHA0_HOST)


class _PadReader(nn.Module):
    """A zero pad read by a perceptron: a conv over a lane zero-padded in channels and in width, then a pool lane and
    a linear head."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.conv = nn.Conv2d(6, 6, 3, padding=(1, 0))
        self.pool = nn.AvgPool2d(2)
        self.head = nn.Linear(6 * 4 * 4, 5)

    def forward(self, x):
        s = torch.relu(self.stem(x))
        y = torch.relu(self.conv(F.pad(s, (1, 1, 0, 0, 0, 2))))
        return self.head(self.pool(y).flatten(1))


class TestThePadIsWiringForAPerceptron:
    def test_a_perceptron_reads_a_padded_lane_exactly(self):
        torch.manual_seed(5)
        program = build_lane_program(_PadReader(), (3, 8, 8), num_classes=5, platform=lane_platform())
        assert host_ops(program) == [] and _plan(program)["pad"] == 1 and _plan(program)["pool"] == 1
        nf, scm, hcm, golden = _four_values(program, real_inputs(program, 16), axons=256, neurons=64)
        assert_exact(nf, scm)
        assert_exact(hcm, scm)
        assert np.array_equal(golden.numpy(), hcm.reshape(golden.shape).numpy())


# ------------------------------------------------------------------------------------------- 3. the refusals
class _HostShortcut(nn.Module):
    """``b + shortcut(s)`` on a 2 x 1 x 3 input where the shortcut stays a host value: a reflect pad of the width (the
    branch a 1 x 3 conv padded by 2), a constant pad of 1, a BatchNorm the stem's two readers keep on the host;
    ``host_fed``: the branch reads the network input (a segment entry)."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        out = 3 if kind == "nonzero_pad" else 2
        self.stem = nn.Conv2d(2, 2, 1)
        self.branch = (nn.Conv2d(2, out, (1, 3), padding=(0, 2)) if kind == "reflect_pad" else nn.Conv2d(2, out, 1))
        self.bn = nn.BatchNorm2d(2)
        with torch.no_grad():
            self.bn.running_mean.uniform_(-0.5, 0.5)
            self.bn.running_var.uniform_(0.5, 2.0)

    def forward(self, x):
        s = torch.relu(self.stem(x))
        if self.kind == "host_fed":
            return (self.branch(x) + s).flatten(1)
        shortcut = {"reflect_pad": lambda: F.pad(s, (1, 1, 0, 0), mode="reflect"),
                    "nonzero_pad": lambda: F.pad(s, (0, 0, 0, 0, 0, 1), value=1.0),
                    "host_bn": lambda: self.bn(s)}[self.kind]()
        return (self.branch(s) + shortcut).flatten(1)


HOST_KINDS = {"reflect_pad": "'pad'", "nonzero_pad": "'pad'", "host_bn": "'bn'", "host_fed": "reads host values"}
#: The flattened output of each kind.
HOST_OUTPUTS = {"reflect_pad": 10, "nonzero_pad": 9, "host_bn": 6, "host_fed": 6}


def _host_program(kind: str, tmp_path, declared: bool):
    torch.manual_seed(7)
    model = _HostShortcut(kind).eval()
    platform = lane_platform(("L", "E", "P"), activation_bits=4)

    def declare(flow, entries, grids, perceptrons, quantizers, lanes):
        platform["quantizer_state"] = write_state(Path(tmp_path) / "state.json",
                                                  {"branch": {"readout_range": 4.0}})

    return build_lane_program(model, (2, 1, 3), num_classes=HOST_OUTPUTS[kind], platform=platform,
                              state_hook=declare if declared else None)


@pytest.mark.parametrize("kind", sorted(HOST_KINDS))
class TestAnAddThatStaysOnTheHost:
    def test_without_a_declared_range_the_add_stays_on_the_host_unchanged(self, kind, tmp_path):
        program = _host_program(kind, tmp_path, declared=False)
        assert _plan(program)["add"] == 0 and ADD_OP in host_ops(program)

    def test_a_declared_range_on_its_holder_is_refused_by_name(self, kind, tmp_path):
        from mimarsinan.mapping.lanes.declared_lanes import HostSumRangeError

        with pytest.raises(HostSumRangeError, match="'branch'") as raised:
            _host_program(kind, tmp_path, declared=True)
        assert "'add'" in str(raised.value) and HOST_KINDS[kind] in str(raised.value)
