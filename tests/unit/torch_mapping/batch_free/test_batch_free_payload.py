"""The host payload of a batch-free value (``mapping/support/batch_free_value.py``) and its bound slot in a
``ComputeAdapter``: computed at every call from its own state (gradients flow), a constant in the deployed program."""

from __future__ import annotations

import operator

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.support.call_template import BoundArg, SourceArg, partition_call
from mimarsinan.mapping.support.compute_modules import ComputeAdapter


def _payload():
    from mimarsinan.mapping.support import batch_free_value

    return batch_free_value


class _Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.arange(6.0, dtype=torch.float64).view(1, 2, 3))
        self.drop = nn.Dropout(0.5)

    def forward(self):
        return self.drop(self.table * 2.0)


def test_value_is_its_stored_form():
    """The stored form of a bound tensor: a leading singleton axis stripped, any other value as it is."""
    module = _payload()
    stripped = module.BatchFreeValue(_Table(), (2, 3))
    assert torch.equal(stripped(), torch.arange(6.0, dtype=torch.float64).view(2, 3) * 2.0)
    kept = module.BatchFreeValue(_Table(), (1, 2, 3))
    assert tuple(kept().shape) == (1, 2, 3)


def test_value_runs_its_program_in_eval_mode():
    """The converter's semantics are eval mode: a module in the computation (a dropout) never draws at run time."""
    value = _payload().BatchFreeValue(_Table(), (2, 3)).train()
    first, second = value(), value()
    assert torch.equal(first, second) and not value.program.training


def test_adapter_computes_a_module_slot_at_every_call():
    table = _Table()
    adapter = ComputeAdapter(operator.add, bound_tensors=[_payload().BatchFreeValue(table, (2, 3))])
    x = torch.zeros(4, 2, 3, dtype=torch.float64)
    out = adapter(x)
    assert torch.equal(out, (table.table * 2.0).expand(4, 1, 2, 3).reshape(4, 2, 3))
    with torch.no_grad():
        table.table.add_(1.0)
    assert torch.equal(adapter(x)[0], (table.table * 2.0)[0])
    adapter(x).sum().backward()
    assert table.table.grad is not None and torch.equal(table.table.grad, torch.full((1, 2, 3), 8.0,
                                                                                     dtype=torch.float64))


def test_deployed_payload_holds_the_constant():
    module = _payload()
    table = _Table()
    adapter = ComputeAdapter(operator.add, bound_tensors=[module.BatchFreeValue(table, (2, 3))])
    deployed = module.deployed_payload(adapter)
    assert deployed is not adapter
    assert not any(isinstance(m, module.BatchFreeValue) for m in deployed.modules())
    x = torch.ones(2, 2, 3, dtype=torch.float64)
    assert torch.equal(deployed(x), adapter(x))
    held = list(deployed.parameters())
    assert len(held) == 1 and not held[0].requires_grad
    with torch.no_grad():
        table.table.add_(1.0)
    assert not torch.equal(deployed(x), adapter(x))  # a constant of the program it was lowered to


def test_deployed_payload_of_any_other_module_is_itself():
    module = _payload()
    plain = ComputeAdapter(operator.add, bound_tensors=[torch.ones(3)])
    assert module.deployed_payload(plain) is plain
    linear = nn.Linear(2, 2)
    assert module.deployed_payload(linear) is linear


def test_partition_carries_the_computing_module():
    """A bound operand computed at run time: the partition's payloads give the module in its slot."""
    graph = fx.Graph()
    x, b = graph.placeholder("x"), graph.placeholder("b")
    node = graph.call_function(operator.add, (x, b))
    computing = nn.Identity()
    stand_in = torch.zeros(3)

    def classify(arg):
        return SourceArg(arg) if arg is x else BoundArg(stand_in, (3,), computing)

    partition = partition_call(operator.add, node.args, node.kwargs, classify)
    assert partition.bound == [stand_in]
    assert partition.payloads() == [computing]
    plain = partition_call(operator.add, node.args, node.kwargs,
                           lambda arg: SourceArg(arg) if arg is x else BoundArg(stand_in, (3,)))
    assert plain.payloads() == [stand_in]
