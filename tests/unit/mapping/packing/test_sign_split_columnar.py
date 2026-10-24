"""Per-axon weight sign on columnar axon sources (integration of the closure's CL-SG with the coverage unit's W9).

The closure splits a compacted soft core's mixed-sign rows into a W+ then W- pair (``apply_sign_split``) after
compaction and before packing; W9 holds the identity build's axon sources as ``SpikeSourceColumns``, whose entries
read back read only. Each test states the listed form's answer and checks the columnar one equals it: the split's
rows, signs and sources, its logical view, and the identity flush of a segment under per-axon sign, core by core,
with the split core's sources still columnar.
"""

from __future__ import annotations

import numpy as np
import pytest

import mimarsinan.mapping.packing.neural_segment_packing as segment_packing
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.packing.hybrid_segment import _flush_neural_segment
from mimarsinan.mapping.packing.neural_segment_packing import neural_segment_to_soft_core_mapping
from mimarsinan.mapping.packing.softcore import SoftCore
from mimarsinan.mapping.packing.softcore.matrix_placement import SignedRowPlacement
from mimarsinan.mapping.packing.softcore.sign_split import apply_sign_split, logical_view
from mimarsinan.mapping.platform.platform_constraints import PER_AXON
from mimarsinan.mapping.pruning.ir_segmentation import build_ir_consumed_by
from mimarsinan.mapping.support.spike_source_columns import SpikeSourceColumns


def _fields(source) -> tuple:
    return (int(source.core_), int(source.neuron_), bool(source.is_input_), bool(source.is_off_),
            bool(source.is_always_on_))


def _int_matrix(rng, rows: int, cols: int) -> np.ndarray:
    """Integer weights with mixed rows, one single-sign row and one all-zero row."""
    w = rng.integers(-7, 8, size=(rows, cols)).astype(np.float64)
    w[1], w[2] = np.abs(w[1]), 0.0
    return w


def _sources(n: int) -> list:
    """Core outputs, a network input, an off source and the always-on bias row last."""
    return ([SpikeSource(4, i) for i in range(n - 3)] + [SpikeSource(-2, 5, is_input=True),
            SpikeSource(-1, 0, is_off=True), SpikeSource(-3, 0, is_always_on=True)])


@pytest.mark.parametrize("seed", range(6))
def test_a_columnar_soft_core_splits_as_its_listed_twin_and_stays_columnar(seed):
    w = _int_matrix(np.random.default_rng(seed), 9, 5)
    listed = SoftCore(core_matrix=w.copy(), axon_sources=_sources(9), id=0)
    columnar = SoftCore(core_matrix=w.copy(), axon_sources=SpikeSourceColumns.of(_sources(9)), id=0)
    apply_sign_split([listed], PER_AXON)
    apply_sign_split([columnar], PER_AXON)
    assert isinstance(columnar.axon_sources, SpikeSourceColumns)
    assert columnar.get_input_count() == listed.get_input_count() > 9
    assert columnar.compact_keep_rows.tolist() == listed.compact_keep_rows.tolist()
    assert columnar.compact_row_signs.tolist() == listed.compact_row_signs.tolist()
    assert [_fields(s) for s in columnar.axon_sources] == [_fields(s) for s in listed.axon_sources]
    assert np.array_equal(columnar.get_core_matrix(), listed.get_core_matrix())
    (logical_c, sources_c), (logical_l, sources_l) = logical_view(columnar), logical_view(listed)
    assert np.array_equal(logical_c, logical_l) and np.array_equal(logical_c, w)
    assert [_fields(s) for s in sources_c] == [_fields(s) for s in sources_l]


def _segment(rng, *, pruned: bool) -> IRGraph:
    """Two integer cores of one segment: a hidden layer over 6 network inputs and the bias, and an output layer over
    the hidden layer's 5 neurons and the bias (its sources are core entries the identity packer remaps)."""
    w1, w2 = _int_matrix(rng, 7, 5), _int_matrix(rng, 6, 3)
    src1 = np.array([IRSource(node_id=-2, index=i) for i in range(6)] + [IRSource(node_id=-3, index=0)], dtype=object)
    src2 = np.array([IRSource(node_id=0, index=i) for i in range(5)] + [IRSource(node_id=-3, index=0)], dtype=object)
    hidden = NeuralCore(id=0, name="hidden", input_sources=src1, core_matrix=w1, threshold=1.0, latency=0)
    out = NeuralCore(id=1, name="out", input_sources=src2, core_matrix=w2, threshold=1.0, latency=1)
    if pruned:
        hidden.pruned_row_mask = [False, False, True, False, False, False, False]
        hidden.pruned_col_mask = [False, False, False, True, False]
    outputs = np.array([IRSource(node_id=1, index=i) for i in range(3)], dtype=object)
    return IRGraph(nodes=[hidden, out], output_sources=outputs, weight_banks={})


def _flush(graph: IRGraph):
    return _flush_neural_segment(current_neural=list(graph.nodes), consumed_by=build_ir_consumed_by(graph),
                                 shared_pool=[], weight_banks=graph.weight_banks, name="segment", identity=True,
                                 sign_granularity=PER_AXON)


def _rows(stage) -> list:
    hcm = stage.hard_core_mapping
    rows = [stage.input_map, stage.output_map, dict(hcm.neuron_mapping.items()),
            [_fields(s) for s in hcm.output_sources.flatten()]]
    for core in hcm.cores:
        rows.append(([_fields(s) for s in core.axon_sources], core.get_axon_source_spans(), core.axons_per_core,
                     core.neurons_per_core, core.get_core_matrix().tolist(),
                     [getattr(p, "row_signs", np.array([])).tolist() for p in core.matrix_placements]))
    return rows


@pytest.mark.parametrize("pruned", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_the_identity_flush_under_per_axon_sign_is_the_listed_one(seed, pruned, monkeypatch):
    graph = _segment(np.random.default_rng(10 + seed), pruned=pruned)
    columnar, columnar_maps = _flush(graph)
    with monkeypatch.context() as patch:
        patch.setattr(segment_packing, "neural_segment_to_soft_core_mapping",
                      lambda g, banks, columnar=False: neural_segment_to_soft_core_mapping(g, banks))
        listed, listed_maps = _flush(_segment(np.random.default_rng(10 + seed), pruned=pruned))
    cores = columnar.hard_core_mapping.cores
    assert all(isinstance(core.axon_sources, SpikeSourceColumns) for core in cores)
    assert all(isinstance(core.axon_sources, list) for core in listed.hard_core_mapping.cores)
    assert any(isinstance(p, SignedRowPlacement) and bool((np.diff(np.asarray(p.keep_rows)) == 0).any())
               for core in cores for p in core.matrix_placements)  # a mixed row was split into a pair
    assert columnar_maps == listed_maps
    assert _rows(columnar) == _rows(listed)
