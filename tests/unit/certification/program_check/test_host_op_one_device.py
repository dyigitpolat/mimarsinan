"""[U-TT T-2] The static check compares a host op's module state on one device.

A GPU session's deployed program holds its host modules on cuda while the identity program the check builds holds
copies on the CPU; ``torch.equal`` across two devices raises, and the check stopped on every GPU session ("the check
did not run: RuntimeError: Expected all tensors to be on the same device"), so both theorem records were failed by
name (the tuned C5 probe on RC4A, defect D2). The comparison moves both states to the CPU first. A fake device stands
in where CUDA is absent: a state tensor that knows its device, and an equality that refuses two devices as CUDA's
does; the CUDA tests run where CUDA is.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.program_check.report import FAILED
from mimarsinan.certification.program_check.tiles import _same_value
from mimarsinan.mapping.ir import ComputeOp

from unit.certification.program_check import programs as P
from unit.certification.program_check.test_mutants import _failed
from unit.mapping.residency.executed import vehicles as V


class _OnDevice:
    """A state tensor on a named device: ``.cpu()`` gives the real CPU tensor."""

    def __init__(self, tensor: torch.Tensor, device: str):
        self._tensor, self.device = tensor, torch.device(device)

    def detach(self):
        return self

    def cpu(self):
        return self._tensor

    def to(self, device):
        return self._tensor if torch.device(device).type == "cpu" else _OnDevice(self._tensor, str(device))


class _TaggedLinear(nn.Linear):
    """A host module whose state reports ``device_tag`` (``"cpu"`` or a fake accelerator)."""

    device_tag = "cpu"

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.device_tag == "cpu":
            return state
        return type(state)((k, _OnDevice(v, self.device_tag)) for k, v in state.items())


_REAL_EQUAL = torch.equal


def _equal_on_one_device(a, b):
    """``torch.equal`` as a two-device call behaves on CUDA: it refuses tensors on two devices."""
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise RuntimeError("Expected all tensors to be on the same device, but got other on another device")
    return _REAL_EQUAL(a, b)


@pytest.fixture
def strict_equal(monkeypatch):
    monkeypatch.setattr(torch, "equal", _equal_on_one_device)


def _pair(device_tag: str, *, same: bool = True):
    torch.manual_seed(0)
    a = _TaggedLinear(4, 3)
    b = copy.deepcopy(a)
    if not same:
        with torch.no_grad():
            b.weight[0, 0] += 1.0
    b.device_tag = device_tag
    return a, b


class TestTheModuleStateIsComparedOnOneDevice:
    def test_a_module_and_its_copy_on_another_device_are_the_same_value(self, strict_equal):
        a, b = _pair("cuda:0")
        assert _same_value(a, b) is True
        assert _same_value(b, a) is True

    def test_a_different_state_on_another_device_is_another_value(self, strict_equal):
        a, b = _pair("cuda:0", same=False)
        assert _same_value(a, b) is False

    def test_on_one_device_the_reading_is_unchanged(self, strict_equal):
        a, b = _pair("cpu")
        assert _same_value(a, b) is True
        a2, b2 = _pair("cpu", same=False)
        assert _same_value(a2, b2) is False

    def test_a_dtype_difference_reads_as_before(self):
        """Only the device changes: across dtypes the reading stays ``torch.equal``'s on the two states."""
        a = nn.Linear(4, 3)
        b = copy.deepcopy(a).double()
        assert _same_value(a, b) is all(_REAL_EQUAL(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def _split_level_with_a_stateful_host_op():
    """G5's split-level vehicle with host_0 a stateful module (the deployed program's host op shares the IR's)."""
    ir = P.integer_ir(V.split_level_repr())
    op = next(n for n in ir.nodes if isinstance(n, ComputeOp) and n.name == "host_0")
    torch.manual_seed(1)
    op.params = dict(op.params, module=_TaggedLinear(V.IN, V.IN).eval())
    with torch.no_grad():
        op.params["module"].weight.copy_(torch.eye(V.IN))
        op.params["module"].bias.zero_()
    return V.build(ir, 3, None), ir


def _host_stage(program, name="host_0"):
    return next(s for s in program.stages if s.kind == "compute" and s.compute_op.name == name)


def _on(program, device_tag: str, *, perturb: bool = False):
    """A copy of ``program`` whose host_0 module is a copy of the IR's on ``device_tag`` (a GPU session's form)."""
    bad = copy.deepcopy(program)
    stage = _host_stage(bad)
    op = copy.copy(stage.compute_op)
    module = copy.deepcopy(op.params["module"])
    module.device_tag = device_tag
    if perturb:
        with torch.no_grad():
            module.bias[0] += 1.0
    op.params = dict(op.params, module=module)
    stage.compute_op = op
    return bad


class TestTheCheckRunsOnADeviceMixedProgram:
    def test_the_check_runs_and_holds_the_host_op(self, strict_equal):
        program, ir = _split_level_with_a_stateful_host_op()
        report = P.run(_on(program, "cuda:0"), ir, P.decl("streamed_lif"))
        tiles = report.results["tiles"]
        assert tiles.status != FAILED, [v.text() for v in tiles.violations]
        assert "host_op" not in {v.rule for v in tiles.violations}

    def test_another_module_on_another_device_still_fails_by_name(self, strict_equal):
        program, ir = _split_level_with_a_stateful_host_op()
        _failed(P.run(_on(program, "cuda:0", perturb=True), ir, P.decl("streamed_lif")), "tiles", "host_op")


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device: the fake device stands in above")


@cuda
class TestOnCuda:
    def test_a_cuda_module_and_its_cpu_copy_are_the_same_value(self):
        a = nn.Conv2d(3, 8, 3)
        b = copy.deepcopy(a).cuda()
        assert _same_value(a, b) is True
        assert _same_value(b, a) is True

    def test_a_different_cuda_module_is_another_value(self):
        a = nn.Linear(4, 3)
        b = copy.deepcopy(a).cuda()
        with torch.no_grad():
            b.weight[0, 0] += 1.0
        assert _same_value(b, a) is False

    def test_the_check_runs_on_a_program_with_its_host_module_on_cuda(self):
        ir = P.integer_ir(V.split_level_repr())
        op = next(n for n in ir.nodes if isinstance(n, ComputeOp) and n.name == "host_0")
        op.params = dict(op.params, module=nn.Linear(V.IN, V.IN).eval())
        program = V.build(ir, 3, None)
        bad = copy.deepcopy(program)
        stage = _host_stage(bad)
        moved = copy.copy(stage.compute_op)
        moved.params = dict(moved.params, module=copy.deepcopy(op.params["module"]).cuda())
        stage.compute_op = moved
        report = P.run(bad, ir, P.decl("streamed_lif"))
        assert "host_op" not in {v.rule for v in report.results["tiles"].violations}
