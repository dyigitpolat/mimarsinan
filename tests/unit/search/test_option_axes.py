"""Deployment options as decision variables, declared once from the config SSOT."""

import pytest

from mimarsinan.search.option_axes import (
    SECTION_DEPLOYMENT,
    SECTION_PLATFORM,
    OptionAxis,
    build_option_axes,
    decode_option_value,
)


class TestDerivationFromTheConfigRegistry:
    """An axis is DESCRIBED by the configurability SSOT, never by a table here."""

    def test_a_bare_key_takes_the_registrys_own_choices(self):
        (axis,) = build_option_axes(["encoding_layer_placement"])
        assert axis.key == "encoding_layer_placement"
        assert axis.choices == ("subsume", "offload")
        assert axis.section == SECTION_DEPLOYMENT

    def test_the_section_comes_from_the_registry_never_from_a_table_here(self):
        """schedule_policy is a DEPLOYMENT key even though the resolved PLATFORM
        carries it — the registry is the only authority on which document it
        belongs in, and the winner stamp writes it back where the registry says."""
        (axis,) = build_option_axes(["schedule_policy"])
        assert axis.choices == ("pool", "bank_clustered")
        assert axis.section == SECTION_DEPLOYMENT

    def test_a_platform_section_key_is_recognised_as_one(self):
        (axis,) = build_option_axes({"weight_bits": {"bounds": [2, 8]}})
        assert axis.section == SECTION_PLATFORM

    def test_a_numeric_key_takes_the_registrys_bounds(self):
        (axis,) = build_option_axes({"pruning_fraction": {}})
        assert axis.choices == ()
        assert axis.bounds == (0.0, 1.0)
        assert axis.section == SECTION_DEPLOYMENT
        assert not axis.integral

    def test_an_int_key_is_integral(self):
        (axis,) = build_option_axes({"weight_bits": {"bounds": [2, 8]}})
        assert axis.integral
        assert axis.bounds == (2.0, 8.0)
        assert axis.section == SECTION_PLATFORM

    def test_a_declaration_may_narrow_the_choices(self):
        (axis,) = build_option_axes({"schedule_policy": ["bank_clustered"]})
        assert axis.choices == ("bank_clustered",)

    def test_narrowing_to_a_value_the_registry_rejects_raises(self):
        with pytest.raises(ValueError, match="nonsense"):
            build_option_axes({"schedule_policy": ["nonsense"]})

    def test_narrowing_beyond_the_registry_bounds_raises(self):
        with pytest.raises(ValueError, match="bounds"):
            build_option_axes({"pruning_fraction": {"bounds": [0.0, 2.0]}})

    def test_an_unknown_key_raises_naming_it(self):
        with pytest.raises(KeyError, match="not_a_key"):
            build_option_axes(["not_a_key"])

    def test_a_key_the_registry_cannot_describe_raises(self):
        """A JSON/recipe key has neither choices nor bounds: it is not searchable."""
        with pytest.raises(ValueError, match="model_config"):
            build_option_axes(["model_config"])

    def test_declaration_order_is_preserved(self):
        axes = build_option_axes(
            ["weight_bits", "encoding_layer_placement", "schedule_policy"]
        )
        assert [a.key for a in axes] == [
            "weight_bits", "encoding_layer_placement", "schedule_policy",
        ]

    def test_an_empty_declaration_yields_no_axes(self):
        assert build_option_axes(None) == ()
        assert build_option_axes([]) == ()

    def test_a_repeated_key_raises(self):
        with pytest.raises(ValueError, match="twice"):
            build_option_axes(["weight_bits", "weight_bits"])


class TestEncoding:
    """Integrality and categoricality live in decode — the box stays real."""

    def test_a_choice_axis_spans_its_index_range(self):
        (axis,) = build_option_axes(["encoding_layer_placement"])
        assert (axis.lower, axis.upper) == (0.0, 1.0)

    def test_a_choice_index_decodes_to_the_value(self):
        (axis,) = build_option_axes(["encoding_layer_placement"])
        assert decode_option_value(axis, 0.0) == "subsume"
        assert decode_option_value(axis, 1.0) == "offload"

    def test_a_choice_index_is_clipped_not_wrapped(self):
        (axis,) = build_option_axes(["encoding_layer_placement"])
        assert decode_option_value(axis, -5.0) == "subsume"
        assert decode_option_value(axis, 99.0) == "offload"

    def test_a_single_choice_axis_is_a_degenerate_span(self):
        (axis,) = build_option_axes({"schedule_policy": ["pool"]})
        assert (axis.lower, axis.upper) == (0.0, 0.0)
        assert decode_option_value(axis, 0.0) == "pool"

    def test_a_numeric_axis_spans_its_bounds(self):
        (axis,) = build_option_axes({"pruning_fraction": {"bounds": [0.0, 0.5]}})
        assert (axis.lower, axis.upper) == (0.0, 0.5)
        assert decode_option_value(axis, 0.25) == pytest.approx(0.25)

    def test_a_numeric_axis_clips_to_its_bounds(self):
        (axis,) = build_option_axes({"pruning_fraction": {"bounds": [0.0, 0.5]}})
        assert decode_option_value(axis, 0.9) == pytest.approx(0.5)
        assert decode_option_value(axis, -1.0) == pytest.approx(0.0)

    def test_an_integral_axis_decodes_to_an_int(self):
        (axis,) = build_option_axes({"weight_bits": {"bounds": [2, 8]}})
        value = decode_option_value(axis, 4.6)
        assert value == 5
        assert isinstance(value, int)


class TestTheAxisIsSelfDescribing:
    def test_every_axis_declares_a_known_section(self):
        for axis in build_option_axes(
            ["encoding_layer_placement", "schedule_policy", "weight_bits",
             "pruning_fraction"]
        ):
            assert axis.section in (SECTION_PLATFORM, SECTION_DEPLOYMENT)

    def test_an_axis_carries_the_registrys_documentation(self):
        (axis,) = build_option_axes(["encoding_layer_placement"])
        assert axis.label
        assert len(axis.doc) >= 15

    def test_a_hand_built_axis_rejects_declaring_both_shapes(self):
        with pytest.raises(ValueError, match="exactly one"):
            OptionAxis(
                key="weight_bits", section=SECTION_PLATFORM,
                choices=(4, 8), bounds=(2.0, 8.0),
                integral=True, label="Weight Bits", doc="both shapes at once",
            )

    def test_a_hand_built_axis_rejects_declaring_neither(self):
        with pytest.raises(ValueError, match="exactly one"):
            OptionAxis(
                key="weight_bits", section=SECTION_PLATFORM,
                choices=(), bounds=None,
                integral=True, label="Weight Bits", doc="neither shape",
            )
