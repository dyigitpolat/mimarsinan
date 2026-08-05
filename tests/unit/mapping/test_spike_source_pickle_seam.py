"""HardCore pickles axon_sources range-compressed; load rebuilds exact SpikeSources."""

from __future__ import annotations

import pickle
import random

import pytest

from mimarsinan.chip_simulation.nevresim.compile_cache import mapping_connectivity_hash
from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import (
    build_synthetic_mapping,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.code_generation.cpp_chip_model_types import compress_sources_to_spans
from mimarsinan.mapping.export.chip_export import hard_cores_to_chip
from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.support.spike_source_spans import (
    SPIKE_SOURCES_DENSE_TAG,
    SPIKE_SOURCES_SPANS_TAG,
    compress_spike_sources,
    decode_spike_sources_packed,
    encode_spike_sources_packed,
    expand_spike_source_spans,
    span_round_trip_exact,
)


def _fields(s: SpikeSource) -> tuple:
    return (s.core_, s.neuron_, s.is_input_, s.is_off_, s.is_always_on_)


def _assert_sources_equal(got: list, expected: list) -> None:
    assert type(got) is list
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert isinstance(g, SpikeSource)
        assert _fields(g) == _fields(e)


def _canonical_repetitive_sources(n: int) -> list:
    run = n // 4
    return (
        [SpikeSource(-2, i, is_input=True) for i in range(run)]
        + [SpikeSource(7, i, is_input=False, is_off=False) for i in range(run)]
        + [SpikeSource(-3, 0, is_always_on=True) for _ in range(run)]
        + [SpikeSource(-1, 0, is_off=True) for _ in range(n - 3 * run)]
    )


def _non_canonical_sources() -> list:
    """Field patterns compress/expand would silently normalize (cf. chip_export input sources)."""
    return [
        SpikeSource(0, 5, is_off=True),
        SpikeSource(-1, 0, is_input=True, is_off=True),
        SpikeSource(7, 3, is_input=True),
        SpikeSource(-2, 4, is_input=True, is_always_on=True),
        SpikeSource(-3, 2, is_always_on=True),
        SpikeSource(0, 0, is_always_on=True),
    ]


def _repetitive_hardcore(n_axons: int) -> HardCore:
    hc = HardCore(axons_per_core=n_axons, neurons_per_core=4, has_bias_capability=False)
    hc.axon_sources = _canonical_repetitive_sources(n_axons)
    hc.available_axons = 0
    hc.available_neurons = 0
    return hc


class TestPackedEncoding:
    def test_round_trip_all_canonical_kinds_uses_spans(self):
        sources = _canonical_repetitive_sources(64)
        payload = encode_spike_sources_packed(sources)
        assert payload[0] == SPIKE_SOURCES_SPANS_TAG
        _assert_sources_equal(decode_spike_sources_packed(payload), sources)

    def test_round_trip_non_canonical_falls_back_to_dense(self):
        sources = _non_canonical_sources()
        payload = encode_spike_sources_packed(sources)
        assert payload[0] == SPIKE_SOURCES_DENSE_TAG
        _assert_sources_equal(decode_spike_sources_packed(payload), sources)

    def test_canonical_but_fragmented_wiring_prefers_dense(self):
        sources = [SpikeSource(i % 3, (i * 7) % 5) for i in range(64)]
        assert all(span_round_trip_exact(s) for s in sources)
        payload = encode_spike_sources_packed(sources)
        assert payload[0] == SPIKE_SOURCES_DENSE_TAG
        _assert_sources_equal(decode_spike_sources_packed(payload), sources)

    def test_empty_list_round_trips(self):
        payload = encode_spike_sources_packed([])
        got = decode_spike_sources_packed(payload)
        assert type(got) is list and got == []

    def test_unknown_tag_raises(self):
        with pytest.raises(ValueError):
            decode_spike_sources_packed(("spike-sources-bogus-v9", None))

    def test_random_lists_round_trip_exactly(self):
        rng = random.Random(1234)
        for _ in range(20):
            sources = [
                SpikeSource(
                    rng.randint(-4, 4),
                    rng.randint(0, 3),
                    rng.random() < 0.3,
                    rng.random() < 0.3,
                    rng.random() < 0.3,
                )
                for _ in range(rng.randint(0, 60))
            ]
            payload = encode_spike_sources_packed(sources)
            _assert_sources_equal(decode_spike_sources_packed(payload), sources)


class TestSpanExactnessPredicate:
    def test_predicate_matches_compress_expand_round_trip(self):
        rng = random.Random(99)
        for _ in range(500):
            s = SpikeSource(
                rng.randint(-4, 4),
                rng.randint(0, 3),
                rng.random() < 0.3,
                rng.random() < 0.3,
                rng.random() < 0.3,
            )
            expanded = expand_spike_source_spans(compress_spike_sources([s]))[0]
            assert span_round_trip_exact(s) == (_fields(expanded) == _fields(s))


class TestHardCorePickleSeam:
    def test_round_trip_fields_including_non_canonical(self):
        hc = HardCore(axons_per_core=32, neurons_per_core=4, has_bias_capability=False)
        hc.axon_sources = _canonical_repetitive_sources(20) + _non_canonical_sources()
        loaded = pickle.loads(pickle.dumps(hc))
        _assert_sources_equal(loaded.axon_sources, hc.axon_sources)

    def test_pickle_size_sublinear_on_repetitive_wiring(self):
        small = pickle.dumps(_repetitive_hardcore(1024))
        big = pickle.dumps(_repetitive_hardcore(16384))
        assert len(big) <= len(small) + 512
        assert len(big) < 8192

    def test_populated_spans_cache_is_dropped_and_recomputes_equal(self):
        hc = _repetitive_hardcore(64)
        spans_before = hc.get_axon_source_spans()
        assert hc._axon_source_spans is not None
        loaded = pickle.loads(pickle.dumps(hc))
        assert loaded.__dict__["_axon_source_spans"] is None
        assert loaded.get_axon_source_spans() == spans_before

    def test_legacy_raw_list_state_loads_unchanged(self):
        hc = _repetitive_hardcore(64)
        legacy_state = dict(hc.__dict__)
        clone = HardCore.__new__(HardCore)
        clone.__setstate__(legacy_state)
        assert clone.axon_sources is legacy_state["axon_sources"]

    def test_unknown_payload_tag_fails_loud_on_load(self):
        hc = _repetitive_hardcore(64)
        state = dict(hc.__dict__)
        state["axon_sources"] = ("spike-sources-bogus-v9", None)
        clone = HardCore.__new__(HardCore)
        with pytest.raises(ValueError):
            clone.__setstate__(state)


class TestConsumersUnchangedAcrossPickle:
    @pytest.mark.parametrize("preset", ["fragmented_crosscore", "contiguous_input"])
    def test_connectivity_hash_stable(self, preset):
        hcm, _ = build_synthetic_mapping(preset, 5, 16, 8)
        before = mapping_connectivity_hash(hcm)
        loaded = pickle.loads(pickle.dumps(hcm))
        assert mapping_connectivity_hash(loaded) == before

    def test_chip_export_spans_identical_after_reload(self):
        hcm, input_size = build_synthetic_mapping("fragmented_crosscore", 5, 16, 8)
        chip_before = hard_cores_to_chip(
            input_size, hcm, hcm.axons_per_core, hcm.neurons_per_core, 0, float,
        )
        loaded = pickle.loads(pickle.dumps(hcm))
        chip_after = hard_cores_to_chip(
            input_size, loaded, loaded.axons_per_core, loaded.neurons_per_core, 0, float,
        )
        assert len(chip_before.connections) == len(chip_after.connections)
        for con_b, con_a in zip(chip_before.connections, chip_after.connections):
            assert compress_sources_to_spans(con_b.axon_sources) == \
                compress_sources_to_spans(con_a.axon_sources)
        for out_b, out_a in zip(chip_before.output_buffer, chip_after.output_buffer):
            assert _fields(out_b) == _fields(out_a)
