"""IR wiring pickles columnar, not as one object per axon.

Measured (2026-08-07): after the heatmap fix the SCM ir_graph pickle's residual
0.83 GB is dominated by ~9.4M ``IRSource`` objects in ``NeuralCore.input_sources``
(~27.7 B each as objects vs 8 B as two int32 columns). An ``IRSource`` is a pair
of ints, so the columnar form is lossless by construction — pinned here.
"""

import pickle

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir.source import (
    decode_ir_sources,
    encode_ir_sources,
    is_encoded_ir_sources,
)


def _sources(n, shape=None):
    kinds = [-1, -2, -3]
    flat = [
        IRSource(node_id=(kinds[i % 3] if i % 7 == 0 else i % 50), index=i % 13)
        for i in range(n)
    ]
    array = np.array(flat, dtype=object)
    return array.reshape(shape) if shape else array


class TestCodecIsLossless:
    def test_round_trip_preserves_every_source(self):
        array = _sources(200)
        got = decode_ir_sources(encode_ir_sources(array))
        assert got.shape == array.shape
        for a, b in zip(got.flatten(), array.flatten()):
            assert (a.node_id, a.index) == (b.node_id, b.index)
            assert (a.is_off(), a.is_input(), a.is_always_on()) == \
                   (b.is_off(), b.is_input(), b.is_always_on())

    def test_multidimensional_shape_survives(self):
        array = _sources(24, shape=(4, 6))
        got = decode_ir_sources(encode_ir_sources(array))
        assert got.shape == (4, 6)
        assert got[2, 3].node_id == array[2, 3].node_id
        assert got[2, 3].index == array[2, 3].index

    def test_empty_wiring_round_trips(self):
        array = np.array([], dtype=object)
        got = decode_ir_sources(encode_ir_sources(array))
        assert got.size == 0

    def test_non_source_entries_are_refused_not_corrupted(self):
        """A mixed array must fall back rather than silently drop entries."""
        array = np.array([IRSource(1, 0), "not-a-source"], dtype=object)
        encoded = encode_ir_sources(array)
        assert not is_encoded_ir_sources(encoded), "mixed arrays must not encode"


def _core_graph(n_axons, n_cores=1):
    nodes = []
    for c in range(n_cores):
        nodes.append(NeuralCore(
            id=c, name=f"c{c}", input_sources=_sources(n_axons),
            # int8 x 1 col: the matrix contributes 1 B/axon, so the bound
            # below measures WIRING, not weights.
            core_matrix=np.zeros((n_axons, 1), dtype=np.int8),
            threshold=1.0, latency=0,
        ))
    return IRGraph(
        nodes=nodes,
        output_sources=np.array([IRSource(0, 0)], dtype=object),
        weight_banks={},
    )


class TestPickleIsColumnar:
    def test_wiring_costs_far_less_than_object_soup(self):
        small = len(pickle.dumps(_core_graph(200)))
        large = len(pickle.dumps(_core_graph(8200)))
        per_source = (large - small) / 8000
        assert per_source < 12, (   # 1 B matrix + 8 B columnar wiring
            f"{per_source:.1f} B/source — IR wiring is still pickling as "
            f"objects (~27.7 B/source)"
        )

    def test_round_trip_through_pickle_preserves_wiring(self):
        graph = _core_graph(64, n_cores=3)
        got = pickle.loads(pickle.dumps(graph))
        for a, b in zip(got.nodes, graph.nodes):
            fa = a.input_sources.flatten()
            fb = b.input_sources.flatten()
            assert len(fa) == len(fb)
            for x, y in zip(fa, fb):
                assert (x.node_id, x.index) == (y.node_id, y.index)

    def test_legacy_states_still_load(self):
        """A pre-codec pickle carries raw object arrays."""
        core = NeuralCore(
            id=0, name="legacy", input_sources=_sources(4),
            core_matrix=np.zeros((4, 2)), threshold=1.0, latency=0,
        )
        state = dict(core.__dict__)          # untouched, raw object array
        fresh = NeuralCore.__new__(NeuralCore)
        fresh.__setstate__(state)
        assert fresh.input_sources.size == 4
        assert isinstance(fresh.input_sources.flatten()[0], IRSource)
