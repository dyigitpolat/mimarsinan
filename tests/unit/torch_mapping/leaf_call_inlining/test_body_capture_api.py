"""The Dynamo capture leaf-call inlining rests on (``fx_rewrites/body_capture.py``): the private torch API pinned, a
broken or missing API leaving every call opaque by name, and the capture's hygiene (no helper run, no state of the
user's written: module globals, parameters, compiled code).

``torch._dynamo.convert_frame.fullgraph_capture`` is private (torch 2.11). A canary capture fails loudly here when it
changes; a capture that fails or is missing never inlines: the call stays opaque, its reason naming the failure, the
converted model exact, stage 2 refused by name.
"""

from __future__ import annotations

import copy
import inspect

import pytest
import torch
import torch._dynamo
import torch.fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import OPAQUE_BODY
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from operand_completeness.harness import assert_converted_exact, assert_impure_stage_2_refused, quietly

W = 4
SHAPE = (W,)


@torch.fx.wrap
def gelu_scaled(t, k):
    return torch.nn.functional.gelu(t) * k


@torch.fx.wrap
def times_weight(t, w):
    return t * w


class GeluNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(W, W), nn.Linear(W, 3)

    def forward(self, x):
        return self.fc2(gelu_scaled(self.fc1(x), 0.5))


class WeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(W, W), nn.Linear(W, 3)
        self.w = nn.Parameter(torch.linspace(0.5, 2.0, W))

    def forward(self, x):
        return self.fc2(times_weight(self.fc1(x), self.w))


SEEN: list = []


@torch.fx.wrap
def spy(t):
    """Records the type of every value it runs on (a side effect: the capture refuses it, the call stays opaque)."""
    SEEN.append(type(t).__name__)
    return t * 2.0


class SpyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(W, 3)

    def forward(self, x):
        return spy(self.fc(x))


def native(factory) -> nn.Module:
    torch.manual_seed(0)
    return factory().double().eval()


def _reasons(gm, helper):
    return [n.meta.get(OPAQUE_BODY) for n in gm.graph.nodes if n.op == "call_function" and n.target is helper]


# ----------------------------------------------------------------------------- the pinned private API
def test_the_private_capture_api_is_the_one_measured():
    """The names the capture reads, as torch 2.11 has them; a change fails here first."""
    from torch._dynamo import convert_frame, source
    from torch.fx.experimental import _config as fx_config

    parameters = list(inspect.signature(convert_frame.fullgraph_capture).parameters)
    assert parameters[:3] == ["mod", "args", "kwargs"], parameters
    assert callable(getattr(convert_frame.CaptureOutput, "forward_callable", None))
    for name in ("LocalSource", "GlobalSource", "TensorPropertySource", "TensorProperty", "GetItemSource",
                 "AttrSource"):
        assert hasattr(source, name), name
    assert {"SIZE", "STRIDE", "STORAGE_OFFSET"} <= set(source.TensorProperty.__members__)
    assert hasattr(fx_config, "backed_size_oblivious") and hasattr(fx_config, "use_duck_shape")
    for key in ("side_effect_replay_policy", "replay_side_effects", "assume_static_by_default",
                "automatic_dynamic_shapes", "specialize_int", "specialize_float", "capture_scalar_outputs",
                "capture_dynamic_output_shape_ops"):
        assert hasattr(torch._dynamo.config, key), key


def test_a_canary_capture_has_the_measured_form():
    """A helper on a batch-carrying operand: one graph, a size symbol of the operand's axis 0 bound to its extent, a
    gelu and a mul, the literal held constant; inlined, the program is pure."""
    from mimarsinan.torch_mapping.fx_rewrites import body_capture

    assert body_capture.capture_unavailable() is None
    gm = trace_model(native(GeluNet), SHAPE)
    assert _reasons(gm, gelu_scaled) == [] and program_purity(gm).pure, gm.graph
    called = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert torch.nn.functional.gelu in called, gm.graph


# ----------------------------------------------------------------------------- a broken or missing capture
def test_a_missing_capture_api_leaves_the_call_opaque_by_name(monkeypatch):
    from torch._dynamo import convert_frame

    monkeypatch.delattr(convert_frame, "fullgraph_capture")
    gm = trace_model(native(GeluNet), SHAPE)
    reasons = _reasons(gm, gelu_scaled)
    assert len(reasons) == 1 and "the body capture is not available" in reasons[0], reasons
    assert program_purity(gm).impurity is not None
    assert_converted_exact(native(GeluNet), SHAPE, "mvm", batches=(1, 2, 9, 13))
    assert_impure_stage_2_refused(native(GeluNet), SHAPE, "gelu_scaled", "gelu_scaled")


@pytest.mark.parametrize("broken", ["raises", "returns_garbage"])
def test_a_failing_capture_never_inlines(monkeypatch, broken):
    from torch._dynamo import convert_frame

    def raising(*_args, **_kwargs):
        raise RuntimeError("an internal error of the capture")

    def garbage(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(convert_frame, "fullgraph_capture", raising if broken == "raises" else garbage)
    gm = trace_model(native(GeluNet), SHAPE)
    reasons = _reasons(gm, gelu_scaled)
    assert len(reasons) == 1 and reasons[0], reasons
    assert program_purity(gm).impurity is not None
    assert_converted_exact(native(GeluNet), SHAPE, "mvm", batches=(1, 2, 9, 13))


# ----------------------------------------------------------------------------- hygiene
def test_the_capture_never_resets_dynamo(monkeypatch):
    def forbidden():
        raise AssertionError("torch._dynamo.reset() recompiles the user's compiled code")

    monkeypatch.setattr(torch._dynamo, "reset", forbidden)
    gm = trace_model(native(GeluNet), SHAPE)
    assert _reasons(gm, gelu_scaled) == []


def test_a_users_compiled_function_is_not_recompiled_by_a_conversion():
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()

    def user_fn(t):
        return torch.sin(t) + 1.0

    # torch.compile rebinds nn.Module.__init__ on its first use (Dynamo's generation tagging), which the conversion
    # refuses by name (T1, RootCallError): it is put back for the conversion and after the test
    init, setstate = nn.Module.__init__, nn.Module.__setstate__
    flag = "___needs_generation_tag_patch"
    had_flag, old_flag = flag in vars(nn.Module), vars(nn.Module).get(flag)
    try:
        compiled = torch.compile(user_fn, backend=counter)
        x = torch.randn(3, 4)
        compiled(x)
        frames = counter.frame_count
        nn.Module.__init__, nn.Module.__setstate__ = init, setstate
        if flag in vars(nn.Module) and not had_flag:
            delattr(nn.Module, flag)
        flow = quietly(convert_torch_model, native(GeluNet), SHAPE, 3, packaging=MVM_PACKAGING,
                       value_program_stage=True)
        assert flow.conversion_certificate.certified
        compiled(x)
        assert counter.frame_count == frames
    finally:
        nn.Module.__init__, nn.Module.__setstate__ = init, setstate
        if had_flag:
            setattr(nn.Module, flag, old_flag)
        elif flag in vars(nn.Module):
            delattr(nn.Module, flag)


def test_the_helpers_module_globals_and_the_users_parameters_are_left_as_they_were():
    module = inspect.getmodule(times_weight)
    names = set(vars(module))
    model = native(WeightNet)
    flow = quietly(convert_torch_model, copy.deepcopy(model), SHAPE, 3, packaging=MVM_PACKAGING,
                   value_program_stage=True)
    assert flow.conversion_certificate.certified
    gm = trace_model(model, SHAPE)
    assert _reasons(gm, times_weight) == []
    assert set(vars(module)) == names, sorted(set(vars(module)) ^ names)
    for parameter in model.parameters():
        assert not [key for key in vars(parameter) if "dynamo" in key], vars(parameter)


def test_a_conversion_runs_the_helper_on_real_tensors_only():
    """The capture interprets the body; the helper's own Python runs only natively (the seeded runs, the shape
    propagation, the probe), on real tensors of real batches [review round 1: on FX's Proxies too]."""
    SEEN.clear()
    try:
        quietly(convert_torch_model, native(SpyNet), SHAPE, 3, packaging=MVM_PACKAGING, value_program_stage=True)
    except ToolchainRefusal:
        pass
    assert SEEN and set(SEEN) == {"Tensor"}, SEEN
