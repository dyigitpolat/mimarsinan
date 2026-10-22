"""Linear-BN-Linear fusion folds through the shared BatchNorm predicate, follows dtype and leaves the native model intact."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.representability_analyzer import (
    CALL_SITE_WEIGHT_SHARING_REASON,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from . import witnesses as W


def _mlp(bias=True):
    return nn.Sequential(nn.Flatten(), nn.Linear(192, 32, bias=bias), nn.BatchNorm1d(32), nn.Linear(32, 16),
                         nn.ReLU(), nn.Linear(16, W.NC))


class _TokenBnBetweenLinears(nn.Module):
    """Linear on (B, L, C), BatchNorm1d over the L axis, Linear: the BN cannot fold into the first Linear."""

    def __init__(self):
        super().__init__()
        self.l1, self.bn, self.l2 = nn.Linear(8, 8), nn.BatchNorm1d(8), nn.Linear(8, 8)
        self.fc = nn.Linear(64, W.NC)

    def forward(self, x):
        h = self.l2(self.bn(self.l1(x.reshape(x.shape[0], 8, 8))))
        return self.fc(torch.flatten(h, 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [True, False])
def test_fusion_never_mutates_the_native_model(dtype, bias):
    torch.manual_seed(0)
    model = W.fp64_eval_copy(_mlp(bias)).to(dtype)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    W.convert_quietly(model, W.SHAPE, "mvm")
    after = model.state_dict()
    assert after.keys() == before.keys()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor), name


@pytest.mark.parametrize("contract", ["mvm", "spiking"])
@pytest.mark.parametrize("bias", [True, False])
def test_fp64_fused_linear_is_exact(contract, bias):
    torch.manual_seed(0)
    model = W.fp64_eval_copy(_mlp(bias))
    assert W.converted_error(model, W.SHAPE, contract) <= W.PARITY_BOUND


def test_fused_linear_follows_the_source_dtype():
    torch.manual_seed(0)
    gm = normalize_fx_graph(trace_model(W.fp64_eval_copy(_mlp()), W.SHAPE, device="cpu"))
    linears = _live_modules(gm, nn.Linear)
    assert len(linears) == 2
    assert {(m.weight.dtype, m.bias.dtype) for m in linears} == {(torch.float64, torch.float64)}


def _live_modules(gm, cls):
    mods = dict(gm.named_modules())
    return [mods[str(n.target)] for n in gm.graph.nodes
            if n.op == "call_module" and isinstance(mods.get(str(n.target)), cls)]


@pytest.mark.parametrize("contract", ["mvm", "spiking"])
def test_token_axis_bn_between_linears_is_not_fused(contract):
    torch.manual_seed(0)
    model = W.fp64_eval_copy(_TokenBnBetweenLinears())
    gm = normalize_fx_graph(trace_model(model, (1, 8, 8), device="cpu"))
    assert len(_live_modules(gm, nn.BatchNorm1d)) == 1 and len(_live_modules(gm, nn.Linear)) == 3
    assert W.converted_error(model, (1, 8, 8), contract) <= W.PARITY_BOUND


def test_token_axis_bn_between_linears_value_program():
    torch.manual_seed(0)
    assert W.value_program_error(W.fp64_eval_copy(_TokenBnBetweenLinears()), (1, 8, 8)) <= W.PARITY_BOUND


class _TiedSecondLinear(nn.Module):
    """``l2`` is fused with ``l1`` at one call site and called raw at another."""

    def __init__(self):
        super().__init__()
        self.l1, self.bn, self.l2, self.r = nn.Linear(16, 16), nn.BatchNorm1d(16), nn.Linear(16, 16), nn.ReLU()
        self.fc = nn.Linear(16, W.NC)

    def forward(self, x):
        x = torch.flatten(x, 1)[:, :16]
        return self.fc(self.l2(self.bn(self.l1(x))) + self.r(self.l2(x)))


@pytest.mark.parametrize("contract", ["mvm", "spiking"])
def test_fusion_keeps_other_call_sites_of_the_second_linear(contract):
    """A Linear another call site reads is never fused, so that site keeps its weights; D13 then refuses the share."""
    torch.manual_seed(0)
    model = W.fp64_eval_copy(_TiedSecondLinear())
    gm = normalize_fx_graph(trace_model(model, W.SHAPE, device="cpu"))
    assert [str(n.target) for n in gm.graph.nodes if n.op == "call_module"].count("l2") == 2
    x = W.seeded_inputs(W.SHAPE)
    with torch.no_grad():
        assert torch.equal(gm(x), model(x))
    with pytest.raises(RepresentabilityError, match=CALL_SITE_WEIGHT_SHARING_REASON):
        W.convert_quietly(model, W.SHAPE, contract)
