"""Torch's declared code mutated in place (review round 11).

``torch_rebinding`` (``symbolic/imported_code.py``) compared every binding of torch's declared code with the object
imported, by identity. A function mutated in place keeps its identity: swapping ``nn.Linear.forward.__code__`` (or a
closure cell, ``__defaults__`` or ``__kwdefaults__``) changes what declared code runs, and the round-10 snapshot
reported nothing (the round-10 reviewer's ``F1`` group):

- ``CodeSwapLinear``: ``nn.Linear.forward.__code__`` replaced by code that doubles a 2-D real input's output above batch
  8; pure with ``torch_rebinding()`` None, the converted flow exact, the value program wrong by rel 0.5 at batches 9
  and 13 with stage 2 passing;
- ``CodeSwapReluModule``: ``F.relu.__code__`` swapped and an ``nn.ReLU`` after a Linear; the spiking conversion
  certified and wrong by rel 0.5 at batches 9 and 13 with its probe passing.

w0-base is wrong on both, and so is the round-9 tree. Now the snapshot keeps, for every Python function it holds
(namespace bindings, class-dict entries through ``staticmethod``, ``classmethod`` and ``property``, code bindings among
module globals, the function an ``lru_cache`` wrapper calls, and the functions in their closure cells), the identities
of ``__code__``, ``__defaults__``, ``__kwdefaults__`` (and its entries) and each closure cell's contents, and
``torch_rebinding`` names any change.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNBase
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.transformer import Transformer

from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import torch_rebinding

from .harness import CONTRACTS, assert_impure_exact, assert_impure_stage_2_refused, native

BATCHES_1_TO_13 = tuple(range(1, 14))


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)


class CodeSwapLinear(_Base):
    def forward(self, x):
        return torch.relu(self.fc(x))


class CodeSwapReluModule(_Base):
    """``nn.ReLU.forward`` calls ``F.relu``."""

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


def _doubling_linear_forward(self, input):
    """Runs with ``nn.Linear.forward``'s globals (``F``, ``torch``)."""
    out = F.linear(input, self.weight, self.bias)
    return out * 2 if type(input) is torch.Tensor and input.dim() == 2 and input.shape[0] > 8 else out


def _doubling_relu(input, inplace=False):
    """Runs with ``F.relu``'s globals (``torch``)."""
    result = torch.relu(input)
    return result * 2 if type(input) is torch.Tensor and input.dim() == 2 and input.shape[0] > 8 else result


@contextlib.contextmanager
def _swapped(function, code):
    original = function.__code__
    function.__code__ = code
    try:
        yield
    finally:
        function.__code__ = original


CODE_SWAPS = {  # form -> (factory, the function, its new code, the reason)
    "linear_forward": (CodeSwapLinear, nn.Linear.forward, _doubling_linear_forward.__code__,
                       "torch.nn.modules.linear.Linear.forward.__code__ is replaced since import"),
    "relu_under_a_relu_module": (CodeSwapReluModule, F.relu, _doubling_relu.__code__,
                                 "torch.nn.functional.relu.__code__ is replaced since import"),
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(CODE_SWAPS))
def test_torch_code_mutated_in_place_makes_the_program_impure_and_its_conversion_exact(how, contract):
    factory, function, code, reason = CODE_SWAPS[how]
    model = native(factory)
    x = torch.randn(9, 8, dtype=torch.float64)
    with torch.no_grad():
        before = model(x)
    with _swapped(function, code):
        with torch.no_grad():  # the witness: the native model differs at batch 9
            assert not torch.equal(model(x), before)
        assert torch_rebinding() == reason
        flow = assert_impure_exact(model, (8,), contract, "fc", "Linear", batches=BATCHES_1_TO_13)
        assert reason in flow.shape_impurity.reason
        if contract == "mvm":
            assert reason in assert_impure_stage_2_refused(model, (8,), "fc", "Linear")
    assert torch_rebinding() is None


def _cell(function, name):
    return dict(zip(function.__code__.co_freevars, function.__closure__))[name]


@contextlib.contextmanager
def _set(target, attribute, value):
    original = getattr(target, attribute)
    setattr(target, attribute, value)
    try:
        yield
    finally:
        setattr(target, attribute, original)


@contextlib.contextmanager
def _item(mapping, key, value):
    original = mapping[key]
    mapping[key] = value
    try:
        yield
    finally:
        mapping[key] = original


def _same_code_elsewhere(function):
    """The same instructions in another code object."""
    return _set(function, "__code__", function.__code__.replace())


_MAX_POOL = F.max_pool2d
_IF_FALSE = _cell(_MAX_POOL, "if_false").cell_contents
_RSUB = _cell(torch.Tensor.__rsub__, "f").cell_contents  # held by the wrapper's closure only

STATE_CHANGES = {  # the state -> (the change, the reason)
    "code": (lambda: _same_code_elsewhere(F.relu), "torch.nn.functional.relu.__code__ is replaced since import"),
    "defaults": (lambda: _set(F.softmax, "__defaults__", (None, 3, torch.float32)),
                 "torch.nn.functional.softmax.__defaults__ is replaced since import"),
    "kwdefaults": (lambda: _set(torch.meshgrid, "__kwdefaults__", {"indexing": "ij"}),
                   "torch.meshgrid.__kwdefaults__ is replaced since import"),
    "kwdefaults_entry": (lambda: _item(torch.meshgrid.__kwdefaults__, "indexing", "ij"),
                         "torch.meshgrid.__kwdefaults__ is replaced since import"),
    "closure_cell": (lambda: _set(_cell(_MAX_POOL, "if_false"), "cell_contents", lambda *a, **k: _IF_FALSE(*a, **k)),
                     "torch.nn.functional.max_pool2d's closure cell 'if_false' is replaced since import"),
    "function_in_a_closure_cell": (lambda: _same_code_elsewhere(_RSUB),
                                   "torch.Tensor.__rsub__ (closure cell 'f').__code__ is replaced since import"),
    "property_accessor": (lambda: _same_code_elsewhere(vars(RNNBase)["all_weights"].fget),
                          "torch.nn.modules.rnn.RNNBase.all_weights.fget.__code__ is replaced since import"),
    "staticmethod": (lambda: _same_code_elsewhere(vars(Transformer)["generate_square_subsequent_mask"].__func__),
                     "torch.nn.modules.transformer.Transformer.generate_square_subsequent_mask.__code__ is replaced "
                     "since import"),
    "classmethod": (lambda: _same_code_elsewhere(vars(Embedding)["from_pretrained"].__func__),
                    "torch.nn.modules.sparse.Embedding.from_pretrained.__code__ is replaced since import"),
    "cache_wrapper": (lambda: _same_code_elsewhere(torch.overrides.get_default_nowrap_functions.__wrapped__),
                      "torch.overrides.get_default_nowrap_functions.__wrapped__.__code__ is replaced since import"),
}


@pytest.mark.parametrize("state", sorted(STATE_CHANGES))
def test_each_mutable_state_of_a_torch_function_is_compared(state):
    change, reason = STATE_CHANGES[state]
    assert torch_rebinding() is None
    with change():
        assert torch_rebinding() == reason
    assert torch_rebinding() is None
