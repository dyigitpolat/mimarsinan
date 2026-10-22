"""Tuple-output leaves: a module call whose traced output is a tuple carries the one element its users read.

The attention leaf's ``output_index`` pattern, generalized: an LSTM or GRU whose sequence output is read converts as
the host op of its own call with ``output_index`` 0, so its identity value program carries a tensor (on the base the
host op returned the tuple and stage 2 refused it, ``NonTensorHostValueError``). A program reading two elements, or an
element that is itself a tuple, keeps the generic path it had (a host op of the tuple and ``getitem`` host ops).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_value_program_exact,
    native,
)

T, F, H = 6, 8, 12


class Recurrent(nn.Module):
    def __init__(self, cell=nn.LSTM, **kw):
        super().__init__()
        self.rnn = cell(F, H, batch_first=True, **kw)
        self.head = nn.Linear(H * (2 if kw.get("bidirectional") else 1), 10)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y.mean(dim=1))


class LastState(nn.Module):
    """Reads the final hidden state of the ``(h, c)`` pair: an element that is itself a tuple."""

    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(F, H, batch_first=True)
        self.head = nn.Linear(H, 10)

    def forward(self, x):
        _, (h, _c) = self.rnn(x)
        return self.head(h[0])


class BothOutputs(nn.Module):
    """Reads the sequence output and the final state: two elements."""

    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(F, H, batch_first=True)
        self.head = nn.Linear(H, 10)

    def forward(self, x):
        y, h = self.rnn(x)
        return self.head(y[:, -1] + h[0])


CASES = {
    "lstm": lambda: Recurrent(nn.LSTM),
    "gru": lambda: Recurrent(nn.GRU),
    "rnn_tanh": lambda: Recurrent(nn.RNN),
    "lstm_2layer_bidirectional": lambda: Recurrent(nn.LSTM, num_layers=2, bidirectional=True),
}


@pytest.mark.parametrize("key", sorted(CASES))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_recurrent_leaf_converts_exactly(key, contract):
    flow = assert_converted_exact(native(CASES[key]), (T, F), contract)
    rnn = [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper) and m.name == "rnn"]
    assert len(rnn) == 1 and rnn[0].output_index == 0


@pytest.mark.parametrize("key", sorted(CASES))
def test_a_recurrent_leaf_value_program_is_exact(key):
    assert_value_program_exact(native(CASES[key]), (T, F))


@pytest.mark.parametrize("factory", [LastState, BothOutputs], ids=["tuple_element", "two_elements"])
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_program_reading_no_single_tensor_element_keeps_the_generic_path(factory, contract):
    flow = assert_converted_exact(native(factory), (T, F), contract)
    rnn = [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper) and m.name == "rnn"]
    assert len(rnn) == 1 and rnn[0].output_index is None


# G1 review round 1: a getitem reads a tuple element only when its producer's traced value is a tuple. ``h[-1]`` on the
# h_n a GRU or RNN carries (output index 1) is an ordinary tensor index, as on the base (spiking certified exact there);
# so is ``o[0]`` on the sequence output an attention leaf carries (on the base it aliased the whole output).


class HiddenState(nn.Module):
    """``_, h = rnn(x); head(h[index])``: the final hidden state of one layer (and direction), batch on axis 1."""

    def __init__(self, cell, index, **kw):
        super().__init__()
        self.rnn, self.index = cell(F, H, batch_first=True, **kw), index
        self.head = nn.Linear(H, 10)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.head(h[self.index])


class AttentionSample(nn.Module):
    """``o = mha(y, y, y)[0]; o + o[0]``: an integer index into the carried attention output."""

    def __init__(self):
        super().__init__()
        self.a, self.mha, self.head = nn.Linear(F, F), nn.MultiheadAttention(F, 2, batch_first=True), nn.Linear(F, 10)

    def forward(self, x):
        y = self.a(x)
        o = self.mha(y, y, y)[0]
        return self.head(o + o[0])


HIDDEN = {
    "gru_h_last": lambda: HiddenState(nn.GRU, -1),
    "gru_h_0": lambda: HiddenState(nn.GRU, 0),
    "gru_2layer_h_1": lambda: HiddenState(nn.GRU, 1, num_layers=2),
    "gru_2layer_h_0": lambda: HiddenState(nn.GRU, 0, num_layers=2),
    "gru_bidirectional_h_1": lambda: HiddenState(nn.GRU, 1, bidirectional=True),
    "rnn_h_last": lambda: HiddenState(nn.RNN, -1),
    "rnn_2layer_h_last": lambda: HiddenState(nn.RNN, -1, num_layers=2),
    "rnn_2layer_h_1": lambda: HiddenState(nn.RNN, 1, num_layers=2),
}
KEYS = [pytest.param(False, id="key_absent"), pytest.param(True, id="decompose")]


def _convert(model, contract, key, stage_2=False):
    import copy

    from mimarsinan.torch_mapping.converter import convert_torch_model
    from operand_completeness.harness import PACKAGINGS, quietly
    kwargs = {"decompose_leaf_modules": True} if key else {}
    return quietly(convert_torch_model, copy.deepcopy(model), (T, F), 10, packaging=PACKAGINGS[contract],
                   value_program_stage=stage_2, **kwargs).eval()


def _refused_or_exact(model, contract, key, stage_2):
    from mimarsinan.common.refusal import ToolchainRefusal
    from operand_completeness.harness import EXACT, worst_error
    try:
        flow = _convert(model, contract, key, stage_2)
    except ToolchainRefusal as refusal:
        return refusal
    err = worst_error(flow, model, (T, F), batches=(1, 2, 7, 13), kinds=("normal", "uniform"))
    assert err <= EXACT, f"converted (certified {flow.conversion_certificate.certified}) with relative error {err:.3e}"
    return flow


@pytest.mark.parametrize("name", sorted(HIDDEN))
@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_index_into_a_carried_hidden_state_is_a_tensor_index(name, key, contract):
    from operand_completeness.harness import EXACT, worst_error
    model = native(HIDDEN[name])
    flow = _convert(model, contract, key)
    assert flow.conversion_certificate.certified
    err = worst_error(flow, model, (T, F), batches=(1, 2, 7, 13))
    assert err <= EXACT, f"relative error {err:.3e}"


@pytest.mark.parametrize("name", sorted(HIDDEN))
@pytest.mark.parametrize("key", KEYS)
def test_an_index_into_a_carried_hidden_state_never_converts_wrong_at_stage_2(name, key):
    _refused_or_exact(native(HIDDEN[name]), "mvm", key, True)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("contract, stage_2", [("spiking", False), ("mvm", False), ("mvm", True)])
def test_an_index_into_a_carried_attention_output_is_refused_by_name_or_exact(key, contract, stage_2):
    _refused_or_exact(native(AttentionSample), contract, key, stage_2)
