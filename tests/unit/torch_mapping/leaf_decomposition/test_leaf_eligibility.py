"""Leaves the decomposition keeps, and attention leaves whose query and key/value lengths differ (G1 review round 1).

A leaf whose native call computes something its decomposition would not stays the leaf under ``decompose_leaf_modules``:

- a ``TransformerEncoderLayer`` whose activation torch's eval fast path reinterprets: it sets
  ``activation_relu_or_gelu`` by ``isinstance`` and computes exact ReLU or GELU, so an ``nn.GELU(approximate='tanh')``
  or a ReLU or GELU subclass is not the function the slow path (and the decomposition) applies;
- a module whose call runs code its class's ``forward`` does not (``symbolic.module_call.call_reason``): a forward
  bound on the instance, a compiled call, a forward hook.

Kept as the leaf, each converts exactly with the key, as it does with the key absent (before: refused by value parity
under the key). An exact ``nn.GELU()`` layer is still decomposed.

An attention leaf with the key absent whose query length differs from the key/value length (cross-attention, memory
tokens in the key and value) is a host op whose operands do not broadcast; its mapper is not an elementwise op, so the
structural forward runs (before: a loud ``ShapeMismatchError`` at the zero-input probe).
"""

from __future__ import annotations

import copy
import types

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.leaf_decomposition.registry import decompose_leaves
from operand_completeness.harness import CONTRACTS, EXACT, PACKAGINGS, native, quietly, worst_error

N, E = 5, 8


def _convert(model, contract, *, decompose, stage_2=False):
    kwargs = {"decompose_leaf_modules": True} if decompose else {}
    return quietly(convert_torch_model, copy.deepcopy(model), (N, E), 10, packaging=PACKAGINGS[contract],
                   value_program_stage=stage_2, **kwargs).eval()


def _assert_exact(model, contract, *, decompose, stage_2=False):
    flow = _convert(model, contract, decompose=decompose, stage_2=stage_2)
    assert flow.conversion_certificate.certified
    err = worst_error(flow, model, (N, E))
    assert err <= EXACT, f"relative error {err:.3e}"


class _ReLU(nn.ReLU):
    def forward(self, x):
        return super().forward(x) * 2.0


class _GELU(nn.GELU):
    def forward(self, x):
        return super().forward(x) + 0.5


class EncoderLayer(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.inp = nn.Linear(E, E)
        self.layer = nn.TransformerEncoderLayer(E, 2, dim_feedforward=2 * E, dropout=0.0, batch_first=True,
                                                activation=activation)
        self.head = nn.Linear(N * E, 10)

    def forward(self, x):
        return self.head(torch.flatten(self.layer(self.inp(x)), 1))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp, self.mha, self.head = nn.Linear(E, E), nn.MultiheadAttention(E, 2, batch_first=True), \
            nn.Linear(N * E, 10)

    def forward(self, x):
        y = self.inp(x)
        return self.head(torch.flatten(self.mha(y, y, y)[0], 1))


def _bound_forward():
    model = Attention()
    native_forward = nn.MultiheadAttention.forward
    model.mha.forward = types.MethodType(
        lambda self, q, k, v, **kw: (native_forward(self, q, k, v, **kw)[0] * 3.0, None), model.mha)
    return model


KEPT = {
    "encoder_gelu_tanh": lambda: EncoderLayer(nn.GELU(approximate="tanh")),
    "encoder_relu_subclass": lambda: EncoderLayer(_ReLU()),
    "encoder_gelu_subclass": lambda: EncoderLayer(_GELU()),
    "attention_bound_forward": _bound_forward,
}


@pytest.mark.parametrize("name", sorted(KEPT))
def test_a_leaf_whose_call_the_decomposition_does_not_compute_stays_the_leaf(name):
    assert decompose_leaves(native(KEPT[name]), (N, E)).replaced == ()


@pytest.mark.parametrize("name", sorted(KEPT))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_kept_leaf_converts_exactly_with_the_key(name, contract):
    _assert_exact(native(KEPT[name]), contract, decompose=True)


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_an_exact_relu_or_gelu_layer_is_still_decomposed(activation):
    assert decompose_leaves(native(lambda: EncoderLayer(activation)), (N, E)).replaced == ("layer",)
    model = native(lambda: EncoderLayer(nn.GELU() if activation == "gelu" else nn.ReLU()))
    assert decompose_leaves(model, (N, E)).replaced == ("layer",)


class CrossLengths(nn.Module):
    """``mha(y[:, :2], kv, kv)`` with ``kdim = vdim = 6``: query length 2, key/value length N."""

    def __init__(self):
        super().__init__()
        self.inp, self.kv = nn.Linear(E, E), nn.Linear(E, 6)
        self.mha = nn.MultiheadAttention(E, 2, kdim=6, vdim=6, batch_first=True)
        self.head = nn.Linear(2 * E, 10)

    def forward(self, x):
        y, kv = self.inp(x), self.kv(x)
        return self.head(torch.flatten(self.mha(y[:, :2], kv, kv, need_weights=False)[0], 1))


class MemoryTokens(nn.Module):
    """Learned memory tokens concatenated into the key and value: key/value length N + 2."""

    def __init__(self):
        super().__init__()
        self.inp, self.memory = nn.Linear(E, E), nn.Parameter(torch.randn(1, 2, E))
        self.mha, self.head = nn.MultiheadAttention(E, 2, batch_first=True), nn.Linear(N * E, 10)

    def forward(self, x):
        y = self.inp(x)
        kv = torch.cat([self.memory.expand(y.shape[0], -1, -1), y], 1)
        return self.head(torch.flatten(self.mha(y, kv, kv)[0], 1))


@pytest.mark.parametrize("factory", [CrossLengths, MemoryTokens], ids=["cross_lengths", "memory_tokens"])
@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_attention_leaf_with_unequal_lengths_converts_with_the_key_absent(factory, contract):
    _assert_exact(native(factory), contract, decompose=False)


# The SDPA rewrite writes ``1 / sqrt(E)`` (E the query's last extent) as a literal, so it needs E to be a proved static
# extent (``fx_rewrites.rewrite_support.StaticExtents``): an impure program proves none, and its call outside the
# declared set may change E with the batch. Rewritten with the traced E, the program below was certified (the probe's
# batches 1 and 6 are at most 8) and wrong from batch 9.


@torch.fx.wrap
def narrower_above_8(t):
    """Outside the declared operator set: (B, N, 8) up to batch 8, (B, N, 4) above."""
    return t if t.shape[0] <= 8 else t[..., :4]


class ImpureHeadDim(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp, self.head = nn.Linear(E, E), nn.Linear(N, 10)

    def forward(self, x):
        q = narrower_above_8(self.inp(x))
        return self.head(torch.nn.functional.scaled_dot_product_attention(q, q, q).sum(-1))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_attention_whose_head_extent_is_not_proved_keeps_its_native_call(contract):
    model = native(ImpureHeadDim)
    flow = _convert(model, contract, decompose=True)
    err = worst_error(flow, model, (N, E), batches=(1, 2, 3, 9, 13), kinds=("normal", "uniform"))
    assert err <= EXACT, f"relative error {err:.3e}"


class TwoSdpaBlocks(nn.Module):
    """Two attention blocks in sequence, timm style: the second block's query is provable only once the first call,
    which has no declared batch rule, is rewritten (``StaticExtents.invalidate``)."""

    def __init__(self):
        super().__init__()
        self.qkv1, self.qkv2, self.head = nn.Linear(E, 3 * E), nn.Linear(E, 3 * E), nn.Linear(N * E, 10)

    @staticmethod
    def _attend(qkv, x):
        b, n, _ = x.shape
        q, k, v = qkv(x).reshape(b, n, 3, 2, E // 2).permute(2, 0, 3, 1, 4)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, E)

    def forward(self, x):
        y = x + self._attend(self.qkv1, x)
        return self.head(torch.flatten(y + self._attend(self.qkv2, y), 1))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_every_attention_of_a_chain_is_decomposed(contract):
    from mimarsinan.mapping.mapping_utils import ComputeOpMapper
    model = native(TwoSdpaBlocks)
    flow = _convert(model, contract, decompose=True)
    assert flow.conversion_certificate.certified
    err = worst_error(flow, model, (N, E))
    assert err <= EXACT, f"relative error {err:.3e}"
    payloads = [type(m.module).__name__ for m in flow.get_mapper_repr().execution_order()
                if isinstance(m, ComputeOpMapper)]
    assert payloads.count("Contraction") == 4, payloads


class GatedMlpAfterSdpa(nn.Module):
    """A SwiGLU-style MLP after an attention call: its chunk's extent is provable once the call, which has no declared
    batch rule, is rewritten, so the SDPA rewrite runs before the chunk rewrite under the key."""

    def __init__(self):
        super().__init__()
        self.qkv, self.up, self.down = nn.Linear(E, 3 * E), nn.Linear(E, 4 * E), nn.Linear(2 * E, E)
        self.head = nn.Linear(N * E, 10)

    def forward(self, x):
        y = x + TwoSdpaBlocks._attend(self.qkv, x)
        a, b = self.up(y).chunk(2, dim=-1)
        return self.head(torch.flatten(y + self.down(a * torch.nn.functional.silu(b)), 1))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_chunk_after_an_attention_is_rewritten_under_the_key(contract):
    model = native(GatedMlpAfterSdpa)
    flow = _convert(model, contract, decompose=True)
    assert flow.conversion_certificate.certified
    err = worst_error(flow, model, (N, E))
    assert err <= EXACT, f"relative error {err:.3e}"
