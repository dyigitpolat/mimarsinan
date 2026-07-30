"""The parity record comparison judges float closeness through the one tolerance policy."""

import math
from dataclasses import dataclass

from mimarsinan.chip_simulation.parity import compare_segment_records

_FIELDS = (("value", lambda r: r.value),)


@dataclass
class _Record:
    value: object


class TestCompareSegmentRecords:
    def test_equal_floats_are_not_a_diff_under_a_zero_tolerance(self):
        assert compare_segment_records(_Record(1.5), _Record(1.5), _FIELDS) == []

    def test_any_float_difference_is_a_diff_under_a_zero_tolerance(self):
        diffs = compare_segment_records(_Record(1.5), _Record(1.5000001), _FIELDS)
        assert [(d.path, d.expected, d.actual) for d in diffs] == [
            ("value", 1.5, 1.5000001)
        ]

    def test_the_absolute_tolerance_is_honoured_at_its_boundary(self):
        assert compare_segment_records(
            _Record(0.5), _Record(0.375), _FIELDS, atol=0.125
        ) == []
        assert compare_segment_records(
            _Record(0.5), _Record(math.nextafter(0.375, 0.0)), _FIELDS, atol=0.125
        ) != []

    def test_the_relative_tolerance_scales_with_the_larger_magnitude(self):
        assert compare_segment_records(
            _Record(4.0), _Record(3.0), _FIELDS, rtol=0.25
        ) == []
        assert compare_segment_records(
            _Record(4.0), _Record(math.nextafter(3.0, 0.0)), _FIELDS, rtol=0.25
        ) != []

    def test_non_float_fields_are_compared_exactly(self):
        assert compare_segment_records(_Record("a"), _Record("a"), _FIELDS) == []
        assert compare_segment_records(_Record(3), _Record(4), _FIELDS) != []
