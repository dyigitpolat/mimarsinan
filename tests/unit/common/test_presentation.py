"""Display/serialization helpers: safe_float conversion and layer grouping keys."""

import pytest

from mimarsinan.common.presentation import layer_key_from_node_name, safe_float


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


class TestLayerKeyFromNodeName:
    def test_collapses_conv_position_cores(self):
        assert layer_key_from_node_name("conv1_pos3_2_g0") == "conv1"

    def test_collapses_fc_tile_cores(self):
        assert layer_key_from_node_name("fc2_tile_1_0") == "fc2"

    def test_collapses_psum_cores(self):
        assert layer_key_from_node_name("conv1_psum_0_1") == "conv1"

    def test_plain_names_pass_through(self):
        assert layer_key_from_node_name("classifier") == "classifier"
