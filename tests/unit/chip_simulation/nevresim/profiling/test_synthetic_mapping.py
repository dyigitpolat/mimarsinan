"""Tests for synthetic HardCoreMapping factory."""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.nevresim.profiling.mapping_metrics import metrics_from_hardcore_mapping
from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import (
    PRESET_ALIASES,
    build_multi_segment_fanout,
    build_synthetic_mapping,
    compute_input_size,
)


@pytest.mark.parametrize("preset", list(PRESET_ALIASES.keys()))
def test_build_synthetic_mapping_all_presets(preset: str) -> None:
    hcm, input_size = build_synthetic_mapping(preset, core_count=4, axons_per_core=8, neurons_per_core=4)
    assert len(hcm.cores) == 4
    assert hcm.axons_per_core == 8
    assert hcm.neurons_per_core == 4
    assert input_size == compute_input_size(preset, 4, 8)
    assert len(hcm.output_sources) >= 1


def test_fragmented_has_more_spans_than_contiguous() -> None:
    contiguous, _ = build_synthetic_mapping("contiguous_input", 10, 32, 16)
    fragmented, _ = build_synthetic_mapping("fragmented_crosscore", 10, 32, 16)
    m_cont = metrics_from_hardcore_mapping(contiguous)
    m_frag = metrics_from_hardcore_mapping(fragmented)
    assert m_frag.max_spans_per_core >= m_cont.max_spans_per_core
    assert m_frag.total_spans >= m_cont.total_spans


def test_patch_embed_like_input_size_scales_with_cores() -> None:
    _, in_small = build_synthetic_mapping("patch_embed_like", 5, 64, 32)
    _, in_large = build_synthetic_mapping("patch_embed_like", 20, 64, 32)
    assert in_small == 5 * 64
    assert in_large == 20 * 64


def test_multi_segment_fanout_count() -> None:
    segments = build_multi_segment_fanout(3, cores_per_segment=5, axons_per_core=16, neurons_per_core=8)
    assert len(segments) == 3
    for hcm, input_size in segments:
        assert len(hcm.cores) == 5
        assert input_size == 5 * 16
