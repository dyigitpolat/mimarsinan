"""Torch's numpy-compatibility keyword names (review round 16, targeted pass).

Torch's Python argument parser accepts, for every builtin function and tensor method, ``axis=`` for ``dim=``,
``keepdims=`` for ``keepdim=``, ``x=``, ``a=`` or ``x1=`` for ``input=`` and ``x2=`` for ``other=``. The round-16
reviewer's witnesses (``review_r16/scripts/r16_bind.py``, each fed to a Linear, each exact on ``w0-base``) were refused:
the rules read ``dim`` as absent (a full reduction, every axis squeezed, axis 0), torch's runs disagreed, and the next
Linear's static layout was refused (``BatchAxisLayoutError``); ``torch.cat([y, y], axis=1)`` was refused as a
concatenation along the batch axis. Every such call is now read with torch's own names (``binding.canonical_keywords``,
applied once to every node of the traced graph and by ``binding.bound``)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.binding import bound
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity import relative_error
from mimarsinan.torch_mapping.value_parity.value_program import identity_value_program

from .harness import CONTRACTS, EXACT, convert, native

ALL_BATCHES = tuple(range(1, 65))

FORMS = {
    "K01": lambda m, y: m.fc12(torch.cat([y, y], axis=1)),
    "K03": lambda m, y: m.fc3(torch.sum(y.view(-1, 2, 3), axis=1)),
    "K04": lambda m, y: m.fc3(y.view(-1, 2, 3).mean(axis=1, keepdims=True).flatten(1)),
    "K06": lambda m, y: m.fc6(torch.squeeze(y.unsqueeze(1), axis=1)),
    "K42": lambda m, y: m.fc1(torch.std(y, axis=1, keepdims=True, correction=0)),
    "K45": lambda m, y: m.fc2(torch.logsumexp(y.view(-1, 2, 3), axis=2)),
    "K46": lambda m, y: m.fc2(torch.amax(y.view(-1, 2, 3), axis=2)),
    "K53": lambda m, y: m.fc3(torch.select(y.view(-1, 2, 3), axis=1, index=0)),
    "K65": lambda m, y: m.fc6(y.unsqueeze(1).squeeze(axis=1)),
    "K40": lambda m, y: m.fc6(torch.matmul(x1=y, x2=m.W)),
    "K41": lambda m, y: m.fc6(torch.add(x=y, other=y)),
}


def _form(name):
    expression = FORMS[name]

    class Form(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 6)
            self.fc12, self.fc6, self.fc3 = nn.Linear(12, 4), nn.Linear(6, 4), nn.Linear(3, 4)
            self.fc2, self.fc1 = nn.Linear(2, 4), nn.Linear(1, 4)
            self.W = nn.Parameter(torch.randn(6, 6))

        def forward(self, x):
            return expression(self, self.fc(x))

    Form.__name__ = Form.__qualname__ = name
    return Form


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("name", sorted(FORMS))
def test_a_numpy_keyword_name_is_read_as_torch_reads_it(name, contract):
    model = native(_form(name))
    flow = convert(model, (8,), contract)
    for batch in ALL_BATCHES:
        x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            err = relative_error(flow(x.clone()), model(x.clone())).relative_error
        assert err <= EXACT, (batch, err)


@pytest.mark.parametrize("name", sorted(FORMS))
def test_the_value_program_of_a_numpy_keyword_name_is_exact(name):
    model = native(_form(name))
    program = identity_value_program(convert(model, (8,), "mvm"))
    for batch in ALL_BATCHES:
        x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            want = model(x.clone())
            err = relative_error(program(x.clone()).reshape(want.shape), want).relative_error
        assert err <= EXACT, (batch, err)


def test_the_traced_graph_holds_torchs_own_names():
    gm = trace_model(native(_form("K04")), (8,))
    mean = next(n for n in gm.graph.nodes if n.target == "mean")
    assert dict(mean.kwargs) == {"dim": 1, "keepdim": True}, mean.kwargs
    gm = trace_model(native(_form("K40")), (8,))
    matmul = next(n for n in gm.graph.nodes if n.target is torch.matmul)
    assert set(matmul.kwargs) == {"input", "other"}, matmul.kwargs


def test_canonical_keywords_rename_as_torchs_parser_does():
    from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.binding import canonical_keywords

    y = torch.zeros(2, 6)
    assert canonical_keywords("call_function", torch.sum, None, (y,), {"axis": 1}) == {"dim": 1}
    assert canonical_keywords("call_method", "mean", None, (y,), {"axis": 1, "keepdims": True}) == {
        "dim": 1, "keepdim": True}
    assert canonical_keywords("call_function", torch.add, None, (), {"a": y, "other": y}) == {"input": y, "other": y}
    assert canonical_keywords("call_function", torch.matmul, None, (), {"x1": y, "x2": y}) == {"input": y, "other": y}
    # a Python function has its own signature (``F.softmax`` takes no ``axis``); a name torch's own takes is kept
    softmax = torch.nn.functional.softmax
    assert canonical_keywords("call_function", softmax, None, (y,), {"axis": 1}) == {"axis": 1}
    assert canonical_keywords("call_function", torch.sum, None, (y,), {"dim": 1}) == {"dim": 1}
    # both names: torch raises; nothing is renamed
    assert canonical_keywords("call_function", torch.sum, None, (y,), {"dim": 1, "axis": 0}) == {"dim": 1, "axis": 0}


def test_bound_reads_a_numpy_keyword_name():
    y = torch.zeros(2, 6)
    holds = lambda value: isinstance(value, torch.Tensor)  # noqa: E731
    assert bound("call_function", torch.sum, None, (y,), {"axis": 1}, holds) == ((y, 1), {})
    assert bound("call_method", "squeeze", None, (y,), {"axis": 1}, holds) == ((y, 1), {})


@torch.fx.wrap
def _as_numpy(t):
    return t.detach().numpy()


class NumpyReceiver(nn.Module):
    """A method of a value that is not a tensor (a numpy array) keeps its own keyword names."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return _as_numpy(self.fc(x)).sum(axis=1)


def test_a_method_of_a_value_that_is_not_a_tensor_keeps_its_keyword_names():
    gm = trace_model(native(NumpyReceiver), (8,))
    total = next(n for n in gm.graph.nodes if n.target == "sum")
    assert dict(total.kwargs) == {"axis": 1}, total.kwargs
