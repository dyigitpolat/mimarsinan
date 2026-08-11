"""Payload-sizing SSOT: exact byte arithmetic + span-count reuse (W4 stage 2)."""

from __future__ import annotations

import pytest

from mimarsinan.code_generation.cpp_chip_model_types import (
    compress_sources_to_spans,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.deployment_record.build.payload_sizes import (
    core_connectivity_entries,
    params_bytes,
    require_weight_bits,
)
from mimarsinan.mapping.packing.softcore.hard_core import HardCore


class TestParamsBytes:
    @pytest.mark.parametrize(
        ("cells", "bits", "expected"),
        [
            (0, 8, 0),
            (5000, 8, 5000),        # byte-aligned
            (10, 4, 5),             # sub-byte width packs
            (3, 3, 2),              # ceil(9 / 8)
            (7, 1, 1),              # ceil(7 / 8)
            (1, 16, 2),             # multi-byte width
            (2049, 4, 1025),        # ceil(8196 / 8)
        ],
    )
    def test_exact_ceil_arithmetic(self, cells, bits, expected):
        assert params_bytes(cells, bits) == expected

    def test_undeclared_weight_bits_fails_loud(self):
        # The DOCUMENTED None choice: params_bytes is an exact int per the
        # schema, so an undeclared platform width raises instead of guessing.
        with pytest.raises(ValueError, match="weight width"):
            params_bytes(100, None)

    def test_non_positive_weight_bits_fails_loud(self):
        with pytest.raises(ValueError, match="positive"):
            params_bytes(100, 0)

    def test_negative_cells_fail_loud(self):
        with pytest.raises(ValueError, match="cells_used"):
            params_bytes(-1, 8)

    def test_require_weight_bits_normalizes_to_int(self):
        assert require_weight_bits(8.0) == 8
        with pytest.raises(ValueError):
            require_weight_bits(None)


def _mixed_source_core() -> HardCore:
    """A hard core whose axon sources span all kinds + a discontinuity."""
    core = HardCore(axons_per_core=12, neurons_per_core=4)
    core.axon_sources = (
        [SpikeSource(0, i) for i in range(3)]           # contiguous core run
        + [SpikeSource(0, 7)]                            # break in stride
        + [SpikeSource(-2, i, is_input=True) for i in range(4)]
        + [SpikeSource(-3, 0, is_always_on=True)]
        + [SpikeSource(-1, 0, is_off=True) for _ in range(3)]
    )
    core._axon_source_spans = None
    return core


class TestSpanCountReuse:
    def test_span_count_equals_the_codegen_ssot(self):
        core = _mixed_source_core()
        expected = len(compress_sources_to_spans(core.axon_sources))
        assert core_connectivity_entries(core) == expected
        # Sanity on the fixture itself: 5 spans (core run, broken core run,
        # input run, always-on, off run).
        assert expected == 5

    def test_span_count_reads_the_cached_hardcore_spans(self):
        core = _mixed_source_core()
        assert core_connectivity_entries(core) == len(
            core.get_axon_source_spans()
        )
