"""safe_float converts or returns default only on conversion errors; others propagate."""

import pytest

from mimarsinan.common.safe_numeric import safe_float


class TestSafeFloat:
    def test_converts_numeric_values(self):
        assert safe_float("1.5") == 1.5
        assert safe_float(2) == 2.0

    def test_value_error_returns_default(self):
        assert safe_float("not-a-number", default=-1.0) == -1.0

    def test_type_error_returns_default(self):
        assert safe_float(None) is None
        assert safe_float(object(), default=0.0) == 0.0

    def test_overflow_error_returns_default(self):
        assert safe_float(10 ** 1000, default=1.0) == 1.0

    def test_unexpected_errors_propagate(self):
        class Exploding:
            def __float__(self):
                raise RuntimeError("backend exploded")

        with pytest.raises(RuntimeError, match="backend exploded"):
            safe_float(Exploding())
