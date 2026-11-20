"""[U-RSHIFT] The received field node by node: each node kind's declared response to a field, and its named refusal.

``received_shifts`` carries a source's field through each node's own forward: a join passes each part's field (an
unshifted part reads zero), a host average pool passes each window's own share (exact in float64), a maximum passes a
field constant on its windows, a LayerNorm absorbs a field constant over its group; any other response refuses by name.
Nothing is computed where no source exists.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support import received_field
from mimarsinan.mapping.support.received_field import ReceivedShiftRefusal, received_shifts

from unit.mapping.received_shift import received_shift_forms as rf


class _PoolRead(nn.Module):
    """A host normalization, then a host pool (``pool``), then optionally a second pool, read by a 1x1 core."""

    def __init__(self, gen, pool: nn.Module, after: nn.Module | None = None):
        super().__init__()
        self.shape = (2, 6, 6)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.bn = nn.BatchNorm2d(3)
        self.pool = pool
        self.after = after if after is not None else nn.Identity()
        self.reader = nn.Sequential(nn.Conv2d(3, 2, 1), nn.ReLU())
        rf.init_(self, gen)

    def forward(self, x):
        return self.reader(self.after(self.pool(self.bn(self.enc(x))))).flatten(1)


def _field_of(model, source_attr: str, value=1.0):
    flow, repr_ = rf.convert(model)
    source = rf.op_named(repr_, source_attr)
    entries = received_shifts(repr_, {source: torch.tensor(value)})
    return repr_, entries


def _reader_field(entries, prefix: str) -> torch.Tensor:
    (node, fields, _own), = [e for e in entries.values() if str(e[0].name).startswith(prefix)]
    (field,) = fields.values()
    return field[0]


def test_a_padded_average_pool_counting_its_padding_passes_each_windows_share():
    model = _PoolRead(torch.Generator().manual_seed(1), nn.AvgPool2d(3, 1, 1))
    _repr, entries = _field_of(model, "bn", 0.75)
    field = _reader_field(entries, "reader")
    shares = nn.AvgPool2d(3, 1, 1)(torch.ones(1, 3, 6, 6, dtype=torch.float64))[0] * 0.75
    assert torch.equal(field, shares)
    assert sorted({round(v, 12) for v in (field[0] / 0.75).reshape(-1).tolist()}) == [round(4 / 9, 12), round(6 / 9, 12),
                                                                                      1.0]


def test_a_full_window_passes_the_field_bit_for_bit():
    model = _PoolRead(torch.Generator().manual_seed(1), nn.AvgPool2d(2))
    _repr, entries = _field_of(model, "bn", 0.1)
    assert torch.equal(_reader_field(entries, "reader"), torch.full((3, 3, 3), 0.1, dtype=torch.float32).double())


class _ConstantRead(nn.Module):
    """A host normalization, then a host op that adds a constant, read by a 1x1 core."""

    def __init__(self, gen):
        super().__init__()
        self.shape = (2, 6, 6)
        self.enc = nn.Sequential(nn.Conv2d(2, 3, 1), nn.ReLU())
        self.bn = nn.BatchNorm2d(3)
        self.reader = nn.Sequential(nn.Conv2d(3, 2, 1), nn.ReLU())
        rf.init_(self, gen)

    def forward(self, x):
        return self.reader(self.bn(self.enc(x)) + 0.25).flatten(1)


def test_a_host_op_adding_a_constant_passes_the_field_as_it_is():
    """A linear op's response is its own forward on the field less on zeros: the constant it adds is no field."""
    _repr, entries = _field_of(_ConstantRead(torch.Generator().manual_seed(8)), "bn", 0.5)
    assert bool((_reader_field(entries, "reader") == 0.5).all())


def test_a_join_passes_each_parts_field_and_the_unshifted_part_reads_zero():
    model = rf.Join(torch.Generator().manual_seed(2), 1)
    _repr, entries = _field_of(model, "bn", 0.5)
    field = _reader_field(entries, "reader")
    assert bool((field[:4] == 0.5).all()) and bool((field[4:] == 0).all())


def test_a_maximum_passes_a_field_constant_on_its_windows_and_refuses_one_that_varies():
    constant = _PoolRead(torch.Generator().manual_seed(3), nn.MaxPool2d(3, 1, 1))
    _repr, entries = _field_of(constant, "bn", 0.25)
    assert bool((_reader_field(entries, "reader") == 0.25).all())
    varying = _PoolRead(torch.Generator().manual_seed(3), nn.AvgPool2d(3, 1, 1), nn.MaxPool2d(2))
    with pytest.raises(ReceivedShiftRefusal, match="maximum over a field that varies"):
        _field_of(varying, "bn")


def test_a_layernorm_absorbs_a_constant_field_and_refuses_a_varying_one():
    absorbing = rf.HostJoin(torch.Generator().manual_seed(4), "layernorm")
    flow, repr_ = rf.convert(absorbing)
    ln = next(o for o in rf.ops(repr_) if isinstance(o.module, nn.LayerNorm))
    shape = tuple(ln.input_shapes[0]) if ln.input_shapes else (150,)
    entries = received_shifts(repr_, {rf.op_named(repr_, "bn"): torch.tensor(0.5)}, refused=(refused := {}))
    assert refused and "LayerNorm" in next(iter(refused.values())), (refused, shape)
    with pytest.raises(ReceivedShiftRefusal, match="LayerNorm"):
        received_shifts(repr_, {rf.op_named(repr_, "bn"): torch.tensor(0.5)})
    assert all(not str(e[0].name).startswith("host") for e in entries.values())


def test_a_shifted_op_whose_input_already_carries_a_field_refuses_by_name():
    model = _PoolRead(torch.Generator().manual_seed(5), nn.AvgPool2d(2))
    flow, repr_ = rf.convert(model)
    sources = {rf.op_named(repr_, "bn"): torch.tensor(0.5), rf.op_named(repr_, "pool"): torch.tensor(0.25)}
    with pytest.raises(ReceivedShiftRefusal, match="already carries a field"):
        received_shifts(repr_, sources)


def test_nothing_is_computed_where_no_source_exists(monkeypatch):
    model = _PoolRead(torch.Generator().manual_seed(6), nn.AvgPool2d(2))
    flow, repr_ = rf.convert(model)

    def _no_pass(*_a, **_k):
        raise AssertionError("a shapes pass ran with no source")
    monkeypatch.setattr(received_field, "mapper_output_shapes", _no_pass)
    assert received_shifts(repr_) == {}


def test_the_shapes_pass_moves_no_running_statistic_and_keeps_the_random_state():
    model = _PoolRead(torch.Generator().manual_seed(7), nn.AvgPool2d(2))
    flow, repr_ = rf.convert(model)
    flow.train()
    stats = [b.detach().clone() for b in flow.buffers()]
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    received_shifts(repr_, {rf.op_named(repr_, "bn"): torch.tensor(0.5)})
    assert torch.equal(torch.rand(3), expected)
    assert all(torch.equal(a, b) for a, b in zip(stats, flow.buffers()))
    assert all(m.training for m in flow.modules())
