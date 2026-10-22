"""``decompose_leaf_modules``: attention and recurrent leaves decomposed, projections on the cores, still certified.

With the key absent every leaf stays the host op of its own call (the base program). With the key, an
``nn.MultiheadAttention``, ``nn.TransformerEncoderLayer`` or ``nn.TransformerEncoder`` leaf is replaced before tracing
by its traceable math on its own parameters (three projections, two Contractions, ``softmax``, the output projection),
and a recurrent leaf is lowered at conversion to its input projections and a ``Recurrence`` host op. The converted
model and the MVM identity value program equal the native model with its leaves (the probe's reference is the user's
own model), the projections are perceptrons under MVM, and a boolean activation mask is refused by name.
"""

from __future__ import annotations

import contextlib
import copy
import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mapping_utils import ComputeOpMapper, PerceptronMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.mapping.verification.onchip_majority import count_host_ops
from mimarsinan.models.vit_leaf import LeafVisionTransformer
from mimarsinan.torch_mapping.converter import convert_torch_model, trace_and_convert
from operand_completeness.harness import CONTRACTS, EXACT, PACKAGINGS, native, quietly, worst_error

N, E = 5, 8


def convert(model, shape, contract, *, decompose=True, stage_2=False):
    return quietly(convert_torch_model, copy.deepcopy(model), tuple(shape), 10, packaging=PACKAGINGS[contract],
                   value_program_stage=stage_2, decompose_leaf_modules=decompose).eval()


class Attention(nn.Module):
    def __init__(self, cross=False, **kw):
        super().__init__()
        self.inp, self.act = nn.Linear(E, E), nn.ReLU()
        self.mha = nn.MultiheadAttention(E, 2, batch_first=True, **kw)
        self.head = nn.Linear(N * E, 10)
        self.cross = cross

    def forward(self, x):
        y = self.act(self.inp(x))
        out, _ = self.mha(y, x if self.cross else y, x if self.cross else y)
        return self.head(torch.flatten(out, 1))


class Encoder(nn.Module):
    def __init__(self, norm_first=False, activation="relu"):
        super().__init__()
        layer = nn.TransformerEncoderLayer(E, 2, dim_feedforward=2 * E, dropout=0.0, batch_first=True,
                                           norm_first=norm_first, activation=activation)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(N * E, 10)

    def forward(self, x):
        return self.head(torch.flatten(self.enc(x), 1))


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(E, 2, dim_feedforward=2 * E, dropout=0.0, batch_first=True,
                                                norm_first=True, activation="gelu")
        self.head = nn.Linear(N * E, 10)

    def forward(self, x):
        return self.head(torch.flatten(self.layer(x), 1))


class Recurrent(nn.Module):
    def __init__(self, cell, **kw):
        super().__init__()
        self.rnn = cell(E, 6, batch_first=True, **kw)
        width = 2 * (kw.get("proj_size") or 6) if kw.get("bidirectional") else (kw.get("proj_size") or 6)
        self.head = nn.Linear(width, 10)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y.mean(dim=1))


def _vit():
    return LeafVisionTransformer(image_size=8, patch_size=4, in_channels=3, embed_dim=8, num_heads=2, depth=2,
                                 mlp_ratio=2.0, num_classes=10)


CASES = {
    "mha_self": (Attention, (N, E)),
    "mha_cross": (lambda: Attention(cross=True), (N, E)),
    "mha_no_bias": (lambda: Attention(bias=False), (N, E)),
    "encoder_post_norm": (Encoder, (N, E)),
    "encoder_pre_norm_gelu": (lambda: Encoder(norm_first=True, activation="gelu"), (N, E)),
    "encoder_layer": (EncoderLayer, (N, E)),
    "vit_mha_leaf": (_vit, (3, 8, 8)),
    "lstm": (lambda: Recurrent(nn.LSTM), (N, E)),
    "gru": (lambda: Recurrent(nn.GRU), (N, E)),
    "rnn_relu": (lambda: Recurrent(nn.RNN, nonlinearity="relu"), (N, E)),
    "lstm_2layer_bidirectional": (lambda: Recurrent(nn.LSTM, num_layers=2, bidirectional=True), (N, E)),
    "lstm_projection": (lambda: Recurrent(nn.LSTM, proj_size=4), (N, E)),
    "gru_bidirectional": (lambda: Recurrent(nn.GRU, bidirectional=True), (N, E)),
}


def _payloads(flow):
    return [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper)]


@pytest.mark.parametrize("key", sorted(CASES))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_decomposed_leaf_converts_exactly_against_the_native_leaf(key, contract):
    factory, shape = CASES[key]
    model = native(factory)
    flow = convert(model, shape, contract)
    assert flow.conversion_certificate.certified
    assert worst_error(flow, model, shape) <= EXACT
    kinds = {type(m.module).__name__ for m in _payloads(flow)}
    assert not kinds & {"MultiheadAttention", "TransformerEncoderLayer", "TransformerEncoder", "LSTM", "GRU", "RNN"}


@pytest.mark.parametrize("key", sorted(CASES))
def test_a_decomposed_leaf_passes_stage_2(key):
    factory, shape = CASES[key]
    model = native(factory)
    flow = convert(model, shape, "mvm", stage_2=True)
    certificate = flow.conversion_certificate
    assert certificate.certified and not certificate.missing_stages and len(certificate.stages) == 2


@pytest.mark.parametrize("key", ["mha_self", "encoder_post_norm", "vit_mha_leaf", "lstm"])
def test_the_projections_are_perceptrons_under_mvm(key):
    factory, shape = CASES[key]
    leaf = convert(native(factory), shape, "mvm", decompose=False)
    decomposed = convert(native(factory), shape, "mvm")
    count = lambda flow: sum(isinstance(m, PerceptronMapper) for m in flow.get_mapper_repr().execution_order())
    per_attention = 4 if key != "lstm" else 1
    blocks = {"mha_self": 1, "encoder_post_norm": 2, "vit_mha_leaf": 2, "lstm": 1}[key]
    extra = 2 * blocks if key == "encoder_post_norm" else 0  # the encoder's linear1 and linear2 leave the leaf too
    assert count(decomposed) - count(leaf) == per_attention * blocks + extra


@pytest.mark.parametrize("contract", CONTRACTS)
def test_with_the_key_absent_every_leaf_stays_its_own_host_op(contract):
    for key in ("mha_self", "encoder_post_norm", "lstm"):
        factory, shape = CASES[key]
        flow = convert(native(factory), shape, contract, decompose=False)
        kinds = {type(m.module).__name__ for m in _payloads(flow)}
        assert kinds & {"MultiheadAttention", "TransformerEncoder", "LSTM"} and "Contraction" not in kinds


def test_the_users_model_is_never_changed():
    model = native(Encoder)
    before = {name: (id(p), p.clone()) for name, p in model.named_parameters()}
    modules = dict(model.named_modules())
    convert(model, (N, E), "mvm")
    quietly(trace_and_convert, model, (N, E), packaging=MVM_PACKAGING, decompose_leaf_modules=True)
    assert dict(model.named_modules()).keys() == modules.keys()
    assert all(type(m) is type(modules[name]) for name, m in model.named_modules())
    for name, p in model.named_parameters():
        assert id(p) == before[name][0] and torch.equal(p, before[name][1])


class Masked(nn.Module):
    def __init__(self, sdpa=False):
        super().__init__()
        self.inp = nn.Linear(E, E)
        self.mha = nn.MultiheadAttention(E, 2, batch_first=True)
        self.head = nn.Linear(N * E, 10)
        self.sdpa = sdpa

    def forward(self, x):
        y = self.inp(x)
        mask = y.sum(-1) > 0
        if self.sdpa:
            b, n, _ = y.shape
            q = y.view(b, n, 2, E // 2).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, q, q, attn_mask=mask[:, None, None, :]).transpose(1, 2)
            return self.head(torch.flatten(out, 1))
        return self.head(torch.flatten(self.mha(y, y, y, key_padding_mask=mask)[0], 1))


@pytest.mark.parametrize("sdpa", [False, True], ids=["mha", "sdpa"])
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_boolean_activation_mask_is_refused_by_name(sdpa, contract):
    from mimarsinan.torch_mapping.leaf_decomposition import ActivationMaskError
    with pytest.raises(ActivationMaskError, match="boolean mask computed from activations"):
        convert(native(lambda: Masked(sdpa)), (N, E), contract)


class WeightsRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(E, 2, batch_first=True)
        self.head = nn.Linear(N * E + N * N, 10)

    def forward(self, x):
        out, weights = self.mha(x, x, x)
        return self.head(torch.cat([torch.flatten(out, 1), torch.flatten(weights, 1)], 1))


@pytest.mark.parametrize("factory", [WeightsRead, lambda: Attention(add_bias_kv=True)], ids=["weights", "bias_kv"])
def test_a_leaf_the_decomposition_does_not_compute_stays_the_leaf(factory):
    model = native(factory)
    leaf_outcome = _outcome(lambda: convert(model, (N, E), "mvm", decompose=False))
    assert _outcome(lambda: convert(model, (N, E), "mvm")) == leaf_outcome


def _outcome(call):
    try:
        flow = call()
    except ToolchainRefusal as refusal:
        return type(refusal).__name__
    return sorted(type(m.module).__name__ for m in _payloads(flow))


def test_a_wrong_decomposition_is_refused_by_the_probe(monkeypatch):
    """The probe's reference is the user's model with its leaves: a decomposition that swapped the key and value
    projections would be refused, so the decomposition is inside the certified claim."""
    from mimarsinan.torch_mapping.leaf_decomposition.attention import DecomposedAttention
    original = DecomposedAttention.__init__

    def swapped(self, mha):
        original(self, mha)
        self.k_proj, self.v_proj = self.v_proj, self.k_proj

    monkeypatch.setattr(DecomposedAttention, "__init__", swapped)
    with pytest.raises(ToolchainRefusal, match="value parity REFUSED"):
        convert(native(Attention), (N, E), "mvm")


def _ir(flow):
    with contextlib.redirect_stdout(io.StringIO()):
        return IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
                         allow_coalescing=False, hardware_bias=False).map(flow.get_mapper_repr())


@pytest.mark.parametrize("key,moved", [
    ("mha_self", 4 * N * E * E),
    ("encoder_post_norm", 2 * (4 * N * E * E + 2 * N * E * 2 * E)),
    ("vit_mha_leaf", 2 * 4 * 5 * 8 * 8),
    ("lstm", N * 4 * 6 * E),
    ("gru", N * 3 * 6 * E),
], ids=["mha", "encoder", "vit_mha_leaf", "lstm", "gru"])
def test_host_macs_fall_by_exactly_the_projection_macs(key, moved):
    factory, shape = CASES[key]
    model = native(factory).float()
    found = {}
    for decompose in (False, True):
        flow = quietly(trace_and_convert, model, shape, packaging=MVM_PACKAGING,
                       decompose_leaf_modules=decompose).flow
        est = estimate_onchip_fraction(flow, shape, 10, metric="macs")
        assert count_host_ops(_ir(flow)) == est.host
        found[decompose] = est
    assert found[False].host - found[True].host == moved
    assert found[True].onchip - found[False].onchip == moved
    assert found[True].total == found[False].total
