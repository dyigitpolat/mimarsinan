"""Stage-2 memory (coverage W9): the per-entry structures of the identity value program, each value unchanged.

Stage 2 holds about one ``IRSource`` per IR input-source entry and one ``SpikeSource`` per core axon, and the value
program a gather plan per host op. On the base each source object carried an instance dict (152 B per object in Python
3.10), a gather plan two tuples of Python ints and their tensor copies, and the identity build re-created every source
an identity compaction map reindexed. Now both source types are slotted (a legacy instance-dict pickle still loads),
the gather plan's index arrays are int64 columns its CPU index tensors share, and a source the map keeps at its index
keeps its object.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pytest
import torch

import mimarsinan.code_generation.cpp_chip_model_types as chip_types
import mimarsinan.mapping.ir.source as source_module
from mimarsinan.code_generation.cpp_chip_model_types import SpikeSource
from mimarsinan.mapping.ir.gather_plan import build_gather_plan, gather_inputs_reference
from mimarsinan.mapping.ir.source import IRSource
from mimarsinan.mapping.packing.hybrid_segment_helpers import _apply_reindex_to_ir_sources


# ----------------------------------------------------------------------------- slotted source types
@pytest.mark.parametrize("value", [IRSource(3, 1000), SpikeSource(4, 1000, is_input=False, is_off=False)],
                         ids=["IRSource", "SpikeSource"])
def test_both_source_types_are_slotted(value):
    assert not hasattr(value, "__dict__")
    assert sys.getsizeof(value) <= 72
    with pytest.raises(AttributeError):
        value.extra = 1


def test_slotted_sources_round_trip_through_pickle():
    ir = pickle.loads(pickle.dumps(IRSource(5, 77)))
    assert ir == IRSource(5, 77)
    spike = pickle.loads(pickle.dumps(SpikeSource(-2, 9, is_input=True)))
    assert (spike.core_, spike.neuron_, spike.is_input_, spike.is_off_, spike.is_always_on_) == (-2, 9, True, False,
                                                                                                    False)


def _legacy_pickle(module, name: str, state: dict, monkeypatch) -> bytes:
    """A pickle as written before the slots: an instance of a plain class of the same module and name, its fields in
    an instance dict."""
    legacy = type(name, (), {"__module__": module.__name__, "__qualname__": name})
    instance = legacy()
    instance.__dict__.update(state)
    with monkeypatch.context() as patch:
        patch.setattr(module, name, legacy)
        return pickle.dumps(instance)


def test_a_legacy_instance_dict_pickle_loads_into_the_slotted_types(monkeypatch):
    ir = pickle.loads(_legacy_pickle(source_module, "IRSource", {"node_id": 6, "index": 12}, monkeypatch))
    assert type(ir) is IRSource and ir == IRSource(6, 12)
    state = {"is_input_": False, "is_off_": True, "is_always_on_": False, "core_": -1, "neuron_": 0}
    spike = pickle.loads(_legacy_pickle(chip_types, "SpikeSource", state, monkeypatch))
    assert type(spike) is SpikeSource and spike.is_off_ and spike.core_ == -1


# ----------------------------------------------------------------------------- columnar gather plans
class _Node:
    def __init__(self, sources):
        self.input_sources = np.array(sources, dtype=object)


def test_a_gather_plan_holds_int64_columns_its_cpu_tensors_share():
    sources = [IRSource(7, 2), IRSource(-2, 1), IRSource(4, 0), IRSource(-3, 0), IRSource(7, 5), IRSource(-1, 0),
               IRSource(4, 3)]
    plan = build_gather_plan(_Node(sources))
    assert plan.referenced_node_ids == (7, 4)
    (seven, dst7, src7), (four, dst4, src4) = plan.buffer_groups
    assert (seven, dst7.tolist(), src7.tolist(), four, dst4.tolist(), src4.tolist()) == (7, [0, 4], [2, 5], 4, [2, 6],
                                                                                         [0, 3])
    assert (plan.input_dst.tolist(), plan.input_src.tolist(), plan.on_dst.tolist()) == ([1], [1], [3])
    for array in (plan.input_dst, plan.input_src, plan.on_dst, dst7, src7):
        assert array.dtype == np.int64
    indices = plan._indices(torch.device("cpu"))
    assert indices["buffers"][0][1].data_ptr() == dst7.ctypes.data
    x, buffers = torch.randn(3, 2, dtype=torch.float64), {7: torch.randn(3, 6), 4: torch.randn(3, 4)}
    assert torch.equal(plan.gather(x, buffers, dtype=torch.float64).float(),
                       gather_inputs_reference(_Node(sources), x, buffers))


def test_a_gather_plan_of_no_source_and_of_off_sources_only():
    for sources in ([], [IRSource(-1, 0)] * 3):
        plan = build_gather_plan(_Node(sources))
        assert plan.n_sources == len(sources) and plan.buffer_groups == ()
        assert torch.equal(plan.gather(torch.ones(2, 1), {}, dtype=torch.float32), torch.zeros(2, len(sources)))


# ----------------------------------------------------------------------------- reindexing keeps kept objects
def test_a_source_the_map_keeps_at_its_index_keeps_its_object():
    kept, moved, dropped, other = IRSource(1, 3), IRSource(1, 4), IRSource(1, 5), IRSource(2, 0)
    sources = np.array([kept, moved, dropped, other, IRSource(-2, 8)], dtype=object)
    _apply_reindex_to_ir_sources(sources, {1: {3: 3, 4: 0}})
    assert sources[0] is kept
    assert sources[1] == IRSource(1, 0) and sources[1] is not moved
    assert sources[2] == IRSource(-1, 0)
    assert sources[3] is other and sources[4] == IRSource(-2, 8)
