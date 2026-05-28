"""Tests for mapping span export and SSOT compression."""

from __future__ import annotations

from mimarsinan.code_generation.cpp_chip_model_types import compress_sources_to_spans
from mimarsinan.code_generation.mapping_spans_export import write_mapping_spans_file
from mimarsinan.mapping.export.chip_export import hard_cores_to_chip
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources, expand_spike_source_spans
from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import build_synthetic_mapping


def test_compress_matches_spike_source_spans_ssot() -> None:
    hcm, input_size = build_synthetic_mapping("fragmented_crosscore", 5, 16, 8)
    chip = hard_cores_to_chip(
        input_size, hcm, hcm.axons_per_core, hcm.neurons_per_core, 0, float,
    )
    for con in chip.connections:
        legacy = compress_sources_to_spans(con.axon_sources)
        ssot = compress_spike_sources(con.axon_sources)
        assert len(legacy) == len(ssot)
        assert sum(s.count for s in legacy) == sum(s.length for s in ssot)


def test_span_round_trip() -> None:
    hcm, input_size = build_synthetic_mapping("contiguous_input", 3, 8, 4)
    chip = hard_cores_to_chip(
        input_size, hcm, hcm.axons_per_core, hcm.neurons_per_core, 0, float,
    )
    for con in chip.connections:
        spans = compress_spike_sources(con.axon_sources)
        expanded = expand_spike_source_spans(spans)
        assert len(expanded) == len(con.axon_sources)
        for a, b in zip(con.axon_sources, expanded):
            assert a.core_ == b.core_ and a.neuron_ == b.neuron_
            assert a.is_input_ == b.is_input_ and a.is_off_ == b.is_off_


def test_write_mapping_spans_file(tmp_path) -> None:
    hcm, input_size = build_synthetic_mapping("patch_embed_like", 4, 8, 4)
    chip = hard_cores_to_chip(
        input_size, hcm, hcm.axons_per_core, hcm.neurons_per_core, 0, float,
    )
    out = tmp_path / "chip_spans.txt"
    write_mapping_spans_file(chip, out)
    text = out.read_text()
    first = text.splitlines()[0].split()
    assert first[0] == str(chip.core_count)
