"""Every operand of a module call is carried into the converted call, or the call is refused by name.

Positional operands after the first, keyword operands (masks, initial states) and an
input passed by keyword were dropped silently before; the value probe passed when the
dropped operand did nothing on its random inputs (a padding mask built from zero frames).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter_handlers.refusals import (
    ModuleCallArgumentError,
    TupleOutputIndexError,
)
from mimarsinan.torch_mapping.mapper_graph_fx import GenericOpArgumentError
from mimarsinan.torch_mapping.value_parity import ConversionValueParityError

from .harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_refused_by_name,
    assert_value_program_exact,
    convert,
    native,
)

D, T, HEADS = 6, 5, 2


class PaddedSelfAttention(nn.Module):
    """The review's witness: frames that are all zero are padding, masked out of attention."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, D)
        self.mha = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.head = nn.Linear(D, 3)

    def forward(self, x):
        pad = torch.sum(torch.abs(x), dim=-1) == 0
        h = self.proj(x)
        y, _ = self.mha(h, h, h, key_padding_mask=pad)
        return self.head(y)


class PaddedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.TransformerEncoderLayer(D, HEADS, 16, dropout=0.0, batch_first=True)
        self.head = nn.Linear(D, 3)

    def forward(self, x):
        pad = torch.sum(torch.abs(x), dim=-1) == 0
        return self.head(self.enc(x, src_key_padding_mask=pad))


class MaskedAttention(nn.Module):
    """MultiheadAttention with a mask held by the model: a float parameter or a boolean causal buffer."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(D, D)
        self.mha = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.head = nn.Linear(D, 3)
        if kind == "float_parameter":
            self.mask = nn.Parameter(torch.randn(T, T))
        else:
            self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        h = self.proj(x)
        y, _ = self.mha(h, h, h, attn_mask=self.mask)
        return self.head(y)


class CausalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.TransformerEncoderLayer(D, HEADS, 16, dropout=0.0, batch_first=True)
        self.register_buffer("src_mask", torch.triu(torch.full((T, T), float("-inf")), diagonal=1))
        self.head = nn.Linear(D, 3)

    def forward(self, x):
        return self.head(self.enc(x, src_mask=self.src_mask))


class StatefulLSTM(nn.Module):
    """LSTM whose initial state is computed from the input (activations)."""

    def __init__(self):
        super().__init__()
        self.h_init = nn.Linear(D, 4)
        self.c_init = nn.Linear(D, 4)
        self.lstm = nn.LSTM(D, 4, batch_first=True)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        s = x.mean(dim=1)
        h0 = self.h_init(s).unsqueeze(0)
        c0 = self.c_init(s).unsqueeze(0)
        out, _ = self.lstm(x, (h0, c0))
        return self.head(out[:, -1])


class LearnedStateLSTM(nn.Module):
    """LSTM whose learned initial state is expanded per batch: an operand with no conversion rule."""

    def __init__(self):
        super().__init__()
        self.h0 = nn.Parameter(torch.randn(1, 1, 4))
        self.c0 = nn.Parameter(torch.randn(1, 1, 4))
        self.lstm = nn.LSTM(D, 4, batch_first=True)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        b = x.shape[0]
        out, _ = self.lstm(x, (self.h0.expand(-1, b, -1).contiguous(), self.c0.expand(-1, b, -1).contiguous()))
        return self.head(out[:, -1])


class KeywordInputs(nn.Module):
    """``self.bn(input=self.c(input=x))``: the activation is passed by keyword."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.r = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 4, 3)

    def forward(self, x):
        return self.fc(torch.flatten(self.r(self.bn(input=self.c(input=x))), 1))


class TwoInputModule(nn.Module):
    """A module of two activations (``nn.Bilinear``): the second operand was dropped before."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(D, 4)
        self.b = nn.Linear(D, 5)
        self.bil = nn.Bilinear(4, 5, 3)

    def forward(self, x):
        return self.bil(self.a(x), self.b(x))


class AttentionWeightsOut(nn.Module):
    """Returns the attention weights (MultiheadAttention's second output)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, D)
        self.mha = nn.MultiheadAttention(D, HEADS, batch_first=True)

    def forward(self, x):
        h = self.proj(x)
        _, weights = self.mha(h, h, h)
        return weights.flatten(1)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_padding_mask_from_activations_is_refused_by_name(contract):
    model = native(PaddedSelfAttention)
    assert_refused_by_name(lambda: convert(model, (T, D), contract), ModuleCallArgumentError,
                           "'mha'", "key_padding_mask", "bool")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_encoder_padding_mask_from_activations_is_refused_by_name(contract):
    model = native(PaddedEncoder)
    assert_refused_by_name(lambda: convert(model, (T, D), contract), ModuleCallArgumentError,
                           "'enc'", "src_key_padding_mask")


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("kind", ["float_parameter", "bool_causal_buffer"])
def test_attention_mask_held_by_the_model_is_carried(kind, contract):
    model = native(lambda: MaskedAttention(kind))
    assert_converted_exact(model, (T, D), contract)


@pytest.mark.parametrize("kind", ["float_parameter", "bool_causal_buffer"])
def test_attention_mask_value_program_is_exact(kind):
    assert_value_program_exact(native(lambda: MaskedAttention(kind)), (T, D))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_encoder_source_mask_buffer_is_carried(contract):
    assert_converted_exact(native(CausalEncoder), (T, D), contract)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_lstm_initial_state_from_activations_is_carried_or_refused_by_name(contract):
    """The state operands reach the LSTM; a state whose batch axis is not leading cannot run per sample.

    Review round 2 (E1): a refusal is read by its base type, ``ToolchainRefusal``; the anonymous forward crash
    at the LSTM it was before is now the named ``unsqueeze(0)`` refusal (the state's batch axis is not leading).
    """
    from mimarsinan.common.refusal import ToolchainRefusal

    model = native(StatefulLSTM)
    try:
        flow = convert(model, (T, D), contract)
    except ToolchainRefusal as refusal:
        assert "'lstm'" in str(refusal) or "lstm" in str(getattr(refusal, "first_diverging_node", "")), str(refusal)
        return
    assert_converted_exact(model, (T, D), contract)
    assert flow is not None


@pytest.mark.parametrize("contract", CONTRACTS)
def test_expanded_learned_state_is_refused_by_name(contract):
    model = native(LearnedStateLSTM)
    assert_refused_by_name(lambda: convert(model, (T, D), contract), GenericOpArgumentError, "'lstm'")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_activation_passed_by_keyword_converts(contract):
    assert_converted_exact(native(KeywordInputs), (3, 4, 4), contract)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_second_activation_operand_is_carried(contract):
    assert_converted_exact(native(TwoInputModule), (D,), contract)


def test_second_activation_operand_value_program_is_exact():
    assert_value_program_exact(native(TwoInputModule), (D,))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_attention_weights_output_is_refused_by_name(contract):
    model = native(AttentionWeightsOut)
    assert_refused_by_name(lambda: convert(model, (T, D), contract), TupleOutputIndexError,
                           "'mha'", "index 1")
