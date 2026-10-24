"""Columnar axon sources (coverage W9): every value the object forms hold, at a fraction of their memory.

The identity build (stage 2's value program) holds a core's axon sources as ``SpikeSourceColumns`` and the hard-core
mapping's neuron map per placement (``PlacementNeuronMap``). Each test states the object form's answer and checks the
columnar one equals it: element by element, spans, the packed pickle payload, compaction's and placement's remaps, the
neuron map's items and its missing keys; and, end to end, the identity hybrid mapping of converted models (one pruned)
built columnar against the same build with lists, stage by stage and core by core, and their value programs' outputs
bit for bit.
"""

from __future__ import annotations

import contextlib
import copy
import io
import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn

import mimarsinan.mapping.packing.neural_segment_packing as segment_packing
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.neural_segment_packing import neural_segment_to_soft_core_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.spike_source_columns import (
    ColumnSpikeSource,
    PlacementNeuronMap,
    SpikeSourceColumns,
    dict_lookup,
)
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources, encode_spike_sources_packed
from mimarsinan.torch_mapping.converter import trace_and_convert


def _fields(source) -> tuple:
    return (int(source.core_), int(source.neuron_), bool(source.is_input_), bool(source.is_off_),
            bool(source.is_always_on_))


def _ir_sources(rng, count: int) -> list:
    """Runs of node outputs, network inputs, off and always-on sources, contiguous and not."""
    out, index = [], 0
    while len(out) < count:
        kind, run = rng.integers(0, 4), int(rng.integers(1, 6))
        node = int(rng.integers(0, 3))
        for _ in range(run):
            if kind == 0:
                out.append(IRSource(node, index))
            elif kind == 1:
                out.append(IRSource(-2, index))
            elif kind == 2:
                out.append(IRSource(-1, int(rng.integers(0, 4))))
            else:
                out.append(IRSource(-3, int(rng.integers(0, 4))))
            index += int(rng.integers(1, 3)) if rng.random() < 0.3 else 1
    return out[:count]


def _as_objects(sources) -> list:
    from mimarsinan.mapping.ir.legacy_convert import ir_source_to_spike_source
    return [ir_source_to_spike_source(s) for s in sources]


@pytest.mark.parametrize("seed", range(6))
def test_columns_read_back_the_sources_their_spans_and_their_packed_payload(seed):
    rng = np.random.default_rng(seed)
    ir = _ir_sources(rng, 200)
    objects = _as_objects(ir)
    columns = SpikeSourceColumns.from_ir_sources(np.array(ir, dtype=object))
    assert [_fields(s) for s in columns] == [_fields(s) for s in objects]
    assert [_fields(columns[i]) for i in (0, 57, -1)] == [_fields(objects[i]) for i in (0, 57, -1)]
    assert SpikeSourceColumns.of(objects) == columns
    assert compress_spike_sources(columns) == compress_spike_sources(objects)
    want, got = encode_spike_sources_packed(objects), encode_spike_sources_packed(columns)
    assert want[0] == got[0] and all(
        np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(want[1:], got[1:]))


def test_an_irregular_source_takes_the_dense_payload_as_its_object_does():
    objects = [SpikeSource(-1, 5, is_off=True)] + [SpikeSource(2, i) for i in range(8)]  # off at neuron 5: not exact
    want, got = encode_spike_sources_packed(objects), encode_spike_sources_packed(SpikeSourceColumns.of(objects))
    assert want[0] == got[0] == "spike-sources-dense-v1"
    assert all(np.array_equal(a, b) for a, b in zip(want[1:], got[1:]))


def test_an_entry_is_read_only_and_its_copy_an_ordinary_source():
    columns = SpikeSourceColumns.of([SpikeSource(3, 4), SpikeSource(-2, 1, is_input=True)])
    entry = columns[0]
    assert isinstance(entry, ColumnSpikeSource)
    with pytest.raises(TypeError, match="read only"):
        entry.neuron_ = 9
    for twin in (copy.copy(entry), pickle.loads(pickle.dumps(entry))):
        assert type(twin) is SpikeSource and _fields(twin) == _fields(entry)
        twin.neuron_ = 9
    columns[0] = SpikeSource(5, 6)
    assert _fields(columns[0]) == (5, 6, False, False, False)
    part = columns[0:1]
    part[0] = SpikeSource(7, 7)
    assert _fields(columns[0]) == (5, 6, False, False, False)


def test_compactions_remap_is_the_per_source_reindex():
    rng = np.random.default_rng(3)
    objects = _as_objects(_ir_sources(rng, 120))
    maps = {0: {n: n + 100 for n in range(0, 400, 2)}, 1: {n: n for n in range(400)}}
    want = []
    for s in objects:
        special = s.is_off_ or s.is_input_ or s.is_always_on_
        if special:
            want.append(_fields(s))
        elif s.core_ in maps and s.neuron_ in maps[s.core_]:
            want.append((s.core_, maps[s.core_][s.neuron_], False, False, False))
        else:
            want.append((-1, 0, False, True, False))
    got = SpikeSourceColumns.of(objects).remapped(dict_lookup(maps))
    assert [_fields(s) for s in got] == want


def _placements(rng):
    """Soft cores split into disjoint fragments, placed at running offsets of a few hard cores."""
    rows, offsets = [], [0, 0, 0]
    for soft in range(6):
        size, first = int(rng.integers(3, 9)), 0
        while first < size:
            count, hard = int(rng.integers(1, size - first + 1)), int(rng.integers(0, 3))
            rows.append((soft, first, count, hard, offsets[hard]))
            offsets[hard] += count
            first += count
    return rows


def test_the_placement_map_is_the_per_neuron_dict():
    rng = np.random.default_rng(5)
    placed, flat = PlacementNeuronMap(), {}
    for soft, first, count, hard, hard_first in _placements(rng):
        placed.place(soft, first, count, hard, hard_first)
        for k in range(count):
            flat[(soft, first + k)] = (hard, hard_first + k)
    assert len(placed) == len(flat) and list(placed) == list(flat) and dict(placed.items()) == flat == placed
    keys = list(flat)
    cores, neurons = np.array([k[0] for k in keys]), np.array([k[1] for k in keys])
    hard, hard_neuron, kept = placed.lookup(cores, neurons)
    assert kept.all() and list(zip(hard.tolist(), hard_neuron.tolist())) == [flat[k] for k in keys]
    with pytest.raises(KeyError) as missing:
        placed[(0, 999)]
    assert missing.value.args[0] == (0, 999)
    with pytest.raises(KeyError) as first_missing:
        placed.lookup(np.array([1, 0, 9]), np.array([0, 999, 0]))
    assert first_missing.value.args[0] == (0, 999)
    with pytest.raises(RuntimeError, match="placed twice"):
        placed.place(0, 0, 1, 2, 50)


# ----------------------------------------------------------------------------- end to end: the identity build
class _Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(6, 6, 3, stride=2)
        self.pool, self.head = nn.AdaptiveAvgPool2d(2), nn.Linear(24, 5)

    def forward(self, x):
        y = torch.relu(self.b(torch.relu(self.a(x))))
        return self.head(torch.flatten(self.pool(y), 1) + 0.5)


class _Tokens(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp, self.norm, self.out = nn.Linear(6, 8), nn.LayerNorm(8), nn.Linear(8, 4)

    def forward(self, x):
        y = self.norm(torch.relu(self.inp(x)))
        return self.out(torch.softmax(y, dim=-1) * y).mean(dim=1)


def _ir(model, shape):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = trace_and_convert(model.double().eval(), shape, "cpu", packaging=MVM_PACKAGING).flow
        return IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
                         allow_coalescing=False, hardware_bias=False).map(flow.get_mapper_repr())


def _pruned(ir, seed: int):
    """Masks on every core: a random third of its rows and columns pruned (the last axon kept, one column kept)."""
    rng = np.random.default_rng(seed)
    for core in ir.nodes:
        if isinstance(core, NeuralCore):
            rows, cols = core.get_core_matrix(ir).shape
            core.pruned_row_mask = [bool(rng.random() < 0.33) and r < rows - 1 for r in range(rows)]
            core.pruned_col_mask = [bool(rng.random() < 0.33) and c > 0 for c in range(cols)]
    return ir


def _listed_build(ir, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(segment_packing, "neural_segment_to_soft_core_mapping",
                      lambda graph, banks, columnar=False: neural_segment_to_soft_core_mapping(graph, banks))
        return build_identity_hybrid_mapping(ir_graph=ir)


def _stage_rows(mapping) -> list:
    rows = []
    for stage in mapping.stages:
        row = [stage.kind, stage.name]
        if stage.kind == "neural":
            hcm = stage.hard_core_mapping
            row += [stage.input_map, stage.output_map, dict(hcm.neuron_mapping.items()),
                    hcm.soft_core_placements_per_hard_core, [_fields(s) for s in hcm.output_sources.flatten()]]
            for core in hcm.cores:
                row.append(([_fields(s) for s in core.axon_sources], core.get_axon_source_spans(), core.axons_per_core,
                            core.neurons_per_core, core.latency, core.threshold, core.core_matrix_key(),
                            None if core.hardware_bias is None else core.hardware_bias.tolist()))
        else:
            row.append([(s.node_id, s.index) for s in stage.compute_op.input_sources.flatten()])
        rows.append(row)
    return rows


@pytest.mark.parametrize("case", ["conv", "tokens", "conv_pruned"])
def test_the_columnar_identity_build_is_the_listed_one_and_its_values_are_bit_equal(case, monkeypatch):
    model, shape = (_Conv(), (3, 8, 8)) if case.startswith("conv") else (_Tokens(), (5, 6))
    ir = _ir(model, shape)
    if case.endswith("pruned"):
        ir = _pruned(ir, seed=1)
    columnar, listed = build_identity_hybrid_mapping(ir_graph=ir), _listed_build(ir, monkeypatch)
    assert any(isinstance(core.axon_sources, SpikeSourceColumns)
               for stage in columnar.stages if stage.kind == "neural" for core in stage.hard_core_mapping.cores)
    assert all(isinstance(core.axon_sources, list)
               for stage in listed.stages if stage.kind == "neural" for core in stage.hard_core_mapping.cores)
    assert _stage_rows(columnar) == _stage_rows(listed)
    x = torch.randn(3, *shape, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        got = ValueHybridCoreFlow(columnar, device="cpu", dtype=torch.float64)(x)
        want = ValueHybridCoreFlow(listed, device="cpu", dtype=torch.float64)(x)
    assert torch.equal(got, want)
    # the forward ran the latency pass on both (``ensure_core_latencies``): the columnar read gives the same tiers
    assert [[core.latency for core in stage.hard_core_mapping.cores] for stage in columnar.stages
            if stage.kind == "neural"] == [[core.latency for core in stage.hard_core_mapping.cores]
                                           for stage in listed.stages if stage.kind == "neural"]
    for stage in columnar.stages:
        if stage.kind == "neural":
            for core in stage.hard_core_mapping.cores:
                twin = pickle.loads(pickle.dumps(core))
                assert [_fields(s) for s in twin.axon_sources] == [_fields(s) for s in core.axon_sources]
