"""A torch-function or dispatch mode entered inside ``forward`` (review round 13).

A mode entered with ``with`` inside ``forward`` is active while FX traces the code under it. FX records a method call,
an operator or a submodule call on a Proxy (``y.size(0)``, ``y * b``, ``self.fc(x)``) without dispatching it through
the mode, while natively the mode intercepts it, and the converted program runs every recorded call with no mode. On
the round-12 tree such programs were pure, held a static shape proof and were certified, and wrong at every batch
from 9 (the reviewer's ``r12_mods.py`` and ``r12_float.py``): ModeSizeTileInForward, under a mode that answers
``size(0)`` with ``min(size, 8)``, had the output shape ``(B, 12)`` against the native ``(B, 6)``; ModeSizeInForward
was wrong by rel 0.125 to 0.875; DispatchModeInForward, a dispatch mode doubling ``aten.addmm`` above 8 rows around a
Linear, by rel 0.5 with stage 2 passing. w0-base refused the tile form at its probe and was wrong on the others.

The tracer refuses, by name, a call it records while a dispatch mode is active that was entered inside the traced code
(``_MimarsinanTracer.create_proxy``): it works on aten operators, below what FX records. A torch function called under a
torch-function mode is dispatched through it first, and the mode's own code runs with the mode popped, so FX records
that code as it runs natively (ModeLinearInForward keeps its conversion); an attribute read records no call. A mode
active before the trace began is the model's environment, the impurity rule's (``test_program_purity.py``), and torch's
own modes (``torch.device``) are declared.

[restated in review round 14: round 13 refused every call FX recorded under such a torch-function mode, a pass-through
one too (CountingModeInForward, exact on w0-base and on the round-12 tree). Now each such call whose native form
dispatches through the mode (a tensor method or operator, a module's call) is traced again alone, with the mode and
without it (``symbolic/entered_modes.py``): the same graph keeps the recorded call; another graph, or one FX cannot
trace, is refused by name.]
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.torch_mapping.torch_graph_tracer import TracingError, trace_model

from .harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_impure_exact,
    assert_impure_stage_2_refused,
    convert,
    native,
)

EVERY_BATCH = tuple(range(1, 14))


class SizeCapMode(TorchFunctionMode):
    """Natively answers ``Tensor.size(0)`` with ``min(size, 8)``; no identity or type test."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(func, "__name__", "") == "size" and tuple(args[1:]) == (0,):
            return min(out, 8)
        return out


class ScaleMode(TorchFunctionMode):
    """Scales every ``F.linear`` output by 2 when its input has more than 8 rows (it reads ``.shape`` in the mode)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(func, "__name__", "") == "linear":
            return out * (1 + (args[0].shape[0] > 8))
        return out


class MmScaleDispatch(TorchDispatchMode):
    """Doubles every ``aten.addmm`` result above 8 rows."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if "addmm" in str(func) and out.shape[0] > 8:
            return out * 2
        return out


class CountingMode(TorchFunctionMode):
    """A pass-through mode (the round-13 reviewer's): it counts the calls and returns each one's own result."""

    calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        CountingMode.calls += 1
        return func(*args, **(kwargs or {}))


class DoubleLinearMode(TorchFunctionMode):
    """Doubles every ``F.linear`` result: around a Linear module's call it changes what the call computes."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        return out * 2 if getattr(func, "__name__", "") == "linear" else out


class InForward(nn.Module):
    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        if self.how == "mode_size":
            y = self.fc(x)
            with SizeCapMode():
                b = y.size(0)
            return y * b
        if self.how == "mode_size_tile":
            y = self.fc(x)
            with SizeCapMode():
                b = y.size(0)
            return torch.tile(y, (1, 1 + (b == x.size(0))))
        if self.how == "dispatch_mode":
            with MmScaleDispatch():
                y = self.fc(x)
            return y
        if self.how == "mode_linear":  # F.linear is dispatched through the mode: FX records the mode's code
            with ScaleMode():
                y = F.linear(x, self.fc.weight, self.fc.bias)
            return y
        if self.how == "device":  # torch's own DeviceContext mode
            with torch.device("cpu"):
                y = self.fc(x) + torch.ones(6, dtype=x.dtype)
            return y
        if self.how == "counting":  # the round-13 reviewer's CountingModeInForward
            with CountingMode():
                y = self.fc(x)
            return F.relu(y)
        if self.how == "counting_operator":  # a pass-through mode around an operator on a tensor and a method
            y = self.fc(x)
            with CountingMode():
                z = (y * 2).relu()
            return z
        if self.how == "size_arithmetic":  # round 13's open issue 7: Python arithmetic on a size read before the mode
            b = x.size(0)
            with CountingMode():
                w = b + 1
            return self.fc(x) * w
        if self.how == "double_module":  # the mode changes what the Linear module's call computes
            with DoubleLinearMode():
                y = self.fc(x)
            return y
        raise ValueError(self.how)


_CHANGED = ("FX records calls on Proxies without dispatching them through the mode, so the traced graph would differ "
            "from the native model, so the model is refused")
REFUSALS = {  # form -> the start of the refusal and the reason it names
    "dispatch_mode": ("fc (the FX node being traced, call_module): a dispatch mode of type MmScaleDispatch is active "
                      "while FX traces it, entered inside the traced code; it works on aten operators, below what FX "
                      "records", "so the traced graph would differ from the native model, so the model is refused"),
    "double_module": ("fc (FX node 'fc', call_module): a mode of type DoubleLinearMode is active while FX traces it, "
                      "entered inside the traced code, and it changes what the call computes", _CHANGED),
}
# form -> the call recorded under the mode that runs with it at run time (an impure program): [restated in review round
# 14: refused by name on the round-13 tree] the mode's code reads a traced value (SizeCapMode's ``min(size, 8)``) or
# calls a tensor method on the Proxy, which FX cannot trace, so the call runs at run time with the mode entered
UNDER_MODE = {"mode_size": ("size", "size"), "mode_size_tile": ("size", "size"), "counting_operator": ("mul", "mul")}


@pytest.mark.parametrize("how", sorted(REFUSALS))
def test_a_call_recorded_under_a_mode_entered_in_forward_is_refused_by_the_tracer(how):
    """[restated in review round 14: the message names what the mode does to the call]"""
    start, reason = REFUSALS[how]
    with pytest.raises(TracingError) as excinfo:
        trace_model(native(lambda: InForward(how)), (8,))
    assert isinstance(excinfo.value, ToolchainRefusal)
    assert str(excinfo.value).startswith(start) and str(excinfo.value).endswith(reason), str(excinfo.value)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(REFUSALS))
def test_the_modes_in_forward_are_refused_by_name_under_both_contracts(how, contract):
    with pytest.raises(TracingError):
        convert(native(lambda: InForward(how)), (8,), contract)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(UNDER_MODE))
def test_a_call_fx_cannot_trace_under_the_mode_runs_with_the_mode_at_run_time(how, contract):
    """ModeSizeInForward and ModeSizeTileInForward were certified and wrong from batch 9 on the round-12 tree, refused
    by name on the round-13 tree: each is exact at every batch now, its program impure at the call that runs under
    the mode (``entered_modes.UnderMode``), and its stage 2 refused by name."""
    node, operator = UNDER_MODE[how]
    flow = assert_impure_exact(native(lambda: InForward(how)), (8,), contract, node, operator, batches=EVERY_BATCH)
    assert flow.shape_impurity.reason == "a call outside the declared operator set"


@pytest.mark.parametrize("how", sorted(UNDER_MODE))
def test_the_stage_2_of_a_call_run_under_its_mode_is_refused_by_name(how):
    node, operator = UNDER_MODE[how]
    assert_impure_stage_2_refused(native(lambda: InForward(how)), (8,), node, operator)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_pass_through_mode_around_a_module_call_keeps_its_conversion(contract):
    """CountingModeInForward was refused by name on the round-13 tree (TracingError at ``fc``), exact on w0-base and
    on the round-12 tree: the mode changes nothing the Linear's call computes (its graph traced alone is the same with
    the mode and without it), so the conversion keeps the Linear's perceptron, pure."""
    flow = assert_converted_exact(native(lambda: InForward("counting")), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert flow.shape_impurity is None


def test_a_pass_through_mode_around_a_module_call_passes_stage_2():
    """As on the round-12 tree: the value program of CountingModeInForward is exact (stage 2 PASS)."""
    from mimarsinan.torch_mapping.converter import convert_torch_model

    from .harness import PACKAGINGS, quietly

    flow = quietly(convert_torch_model, native(lambda: InForward("counting")), (8,), 10, packaging=PACKAGINGS["mvm"],
                   value_program_stage=True)
    assert flow.conversion_certificate.certified


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["device", "mode_linear"])
def test_a_torch_mode_and_a_mode_whose_code_fx_records_keep_their_conversion(how, contract):
    """``with torch.device("cpu")`` (torch's own mode) and ScaleMode around ``F.linear`` (dispatched through the mode,
    whose code FX records with the mode popped, as it runs natively): exact at every batch, as on the round-12 tree."""
    assert_converted_exact(native(lambda: InForward(how)), (8,), contract, batches=EVERY_BATCH,
                           kinds=("normal", "uniform"))


def test_a_mode_active_before_the_trace_is_not_refused_by_the_tracer():
    """The model's environment: the impurity rule names it (``test_program_purity.py``), the tracer does not."""
    with SizeCapMode():
        gm = trace_model(native(lambda: InForward("device")), (8,))
    assert any(node.op == "call_module" for node in gm.graph.nodes)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_python_arithmetic_on_a_size_under_a_mode_is_kept(contract):
    """Round 13's open issue 7: the round-13 tracer refused ``b + 1`` under a mode entered in ``forward`` (FX records it
    on a Proxy). Natively Python's integer arithmetic dispatches through no mode, so the call is kept: the program is
    pure, its width folds nothing, and it is exact at every batch."""
    flow = assert_converted_exact(native(lambda: InForward("size_arithmetic")), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert flow.shape_impurity is None
