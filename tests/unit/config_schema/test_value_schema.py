"""The one value schema and the one loop that judges scalars, JSON keys and core-type fields alike."""

import ast
import dataclasses
import inspect
import json

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY, Category, ConfigKeySchema, FieldType
from mimarsinan.config_schema.registry import value_schema as vs
from mimarsinan.config_schema.registry.entries_write import RECIPE_SCHEMA
from mimarsinan.config_schema.registry.value_schema import (
    ANY,
    DeclarationValueError,
    RequiredIff,
    ValueSchema,
    describe,
    domain_errors,
    entry_value_errors,
    judge_declaration,
    nested_schemas,
    schema_of,
    value_errors,
)
from mimarsinan.config_schema.validation import validate_against_registry

_INT_GE_2 = ValueSchema(kind=FieldType.INT, low=2)
_FLOAT_GT_0 = ValueSchema(kind=FieldType.FLOAT, low=0.0, low_exclusive=True)
_FRACTION = ValueSchema(kind=FieldType.FLOAT, low=0.0, high=1.0, high_exclusive=True)
_LAW = ValueSchema(
    kind=FieldType.JSON,
    fields={
        "bits": _INT_GE_2,
        "range": ValueSchema(kind=FieldType.ENUM, options=("calibrated", "fixed")),
        "range_value": _FLOAT_GT_0,
    },
    required=("bits", "range"),
    required_iff=(RequiredIff("range_value", "range", ("fixed",)),),
)


class TestScalars:
    def test_inclusive_lower_bound(self):
        assert value_errors(_INT_GE_2, 2, "k") == []
        assert value_errors(_INT_GE_2, 1, "k") == ["k=1 is below the minimum 2"]

    def test_exclusive_lower_bound(self):
        assert value_errors(_FLOAT_GT_0, 1e-9, "k") == []
        (message,) = value_errors(_FLOAT_GT_0, 0, "k")
        assert message.startswith("k=0 ") and "exclusive" in message

    def test_exclusive_upper_bound(self):
        assert value_errors(_FRACTION, 0.0, "k") == []
        assert value_errors(_FRACTION, 0.999, "k") == []
        (message,) = value_errors(_FRACTION, 1.0, "k")
        assert message.startswith("k=1.0 ") and "exclusive" in message

    def test_bool_is_not_an_int_and_int_is_a_float(self):
        """The type message, exactly: under INT >= 2, True (which Python counts as 1) would also fail the
        bound, so the bound alone cannot show that bool is refused as a type."""
        assert value_errors(_INT_GE_2, True, "k") == ["k must be int, got bool (True)"]
        assert value_errors(_FLOAT_GT_0, 3, "k") == []

    @pytest.mark.parametrize("value", [True, False])
    @pytest.mark.parametrize("schema", [ValueSchema(kind=FieldType.INT, low=0),
                                        ValueSchema(kind=FieldType.INT)], ids=["int>=0", "int"])
    def test_a_bool_is_refused_where_its_integer_would_pass(self, schema, value):
        assert value_errors(schema, int(value), "k") == []
        assert value_errors(schema, value, "k") == [f"k must be int, got bool ({value!r})"]
        assert value_errors(ValueSchema(kind=FieldType.FLOAT, low=0.0), value, "k") == [
            f"k must be float, got bool ({value!r})"]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("schema", [_FLOAT_GT_0, _FRACTION, ValueSchema(kind=FieldType.FLOAT)],
                             ids=["above", "between", "unbounded"])
    def test_a_float_must_be_finite(self, schema, value):
        """Every comparison with NaN is False, so a bound alone would admit it (json.load reads NaN)."""
        assert value_errors(schema, value, "k") == [f"k={value!r} must be a finite number"]

    def test_an_integer_beyond_float_range_is_judged_exactly(self):
        """json.loads reads a 400-digit integer as an int; an int is finite, and the bound check
        compares it exactly (math.isfinite would raise OverflowError on it)."""
        huge = json.loads("1" + "0" * 400)
        assert isinstance(huge, int)
        assert value_errors(ValueSchema(kind=FieldType.FLOAT), huge, "k") == []
        assert value_errors(_FLOAT_GT_0, huge, "k") == []
        (above,) = value_errors(_FRACTION, huge, "k")
        assert above == f"k={huge!r} must be below 1.0 (exclusive bound)"
        (below,) = value_errors(_FLOAT_GT_0, -huge, "k")
        assert below == f"k={-huge!r} must be above 0.0 (exclusive bound)"

    def test_a_registry_float_key_names_an_integer_beyond_float_range(self):
        huge = json.loads("1" + "0" * 400)
        (row,) = validate_against_registry({"lr": huge})
        assert row["key"] == "lr" and "is above the maximum" in row["message"]

    def test_enum_options(self):
        schema = ValueSchema(kind=FieldType.ENUM, options=("a", "b"))
        assert value_errors(schema, "a", "k") == []
        assert value_errors(schema, "c", "k") == [
            "k='c' is not a valid option; choose one of ['a', 'b']"
        ]

    def test_a_reserved_value_refuses_by_name_as_reserved(self):
        """A value the consumer names but does not run is no legal value: it is refused as reserved,
        and the legal set is the options alone."""
        schema = ValueSchema(kind=FieldType.ENUM, options=("a",), reserved=("b", "c"))
        assert value_errors(schema, "a", "k") == []
        assert value_errors(schema, "b", "k") == [
            "k='b' is reserved/not implemented; only 'a' is supported"]
        assert value_errors(schema, "z", "k") == [
            "k='z' is not a valid option; choose one of ['a']"]
        several = ValueSchema(kind=FieldType.ENUM, options=("a", "b"), reserved=("c",))
        assert value_errors(several, "c", "k") == [
            "k='c' is reserved/not implemented; only 'a', 'b' are supported"]

    @pytest.mark.parametrize("options,reserved", [(None, ("b",)), (("a", "b"), ("b",))],
                             ids=["no-legal-set", "reserved-inside-the-legal-set"])
    def test_a_reserved_value_sits_beside_a_closed_legal_set(self, options, reserved):
        with pytest.raises(ValueError, match="reserved"):
            ValueSchema(kind=FieldType.ENUM, options=options, reserved=reserved)

    def test_non_empty_string(self):
        schema = ValueSchema(kind=FieldType.STR, min_length=1)
        assert value_errors(schema, "why", "k") == []
        assert value_errors(schema, "", "k")
        assert value_errors(schema, 3, "k")

    def test_any_accepts_everything(self):
        schema = ValueSchema(kind=ANY)
        for value in (None, 0, "x", [1], {"a": 1}):
            assert value_errors(schema, value, "k") == []


class TestNested:
    def test_a_legal_object_passes(self):
        assert value_errors(_LAW, {"bits": 4, "range": "calibrated"}, "law") == []
        assert value_errors(
            _LAW, {"bits": 4, "range": "fixed", "range_value": 2.5}, "law"
        ) == []

    def test_nested_bound_names_the_path(self):
        (message,) = value_errors(_LAW, {"bits": 1, "range": "calibrated"}, "law")
        assert message.startswith("law.bits=1")

    def test_required_field_missing(self):
        (message,) = value_errors(_LAW, {"range": "calibrated"}, "law")
        assert "law.bits" in message and "required" in message

    def test_conditional_requirement_both_directions(self):
        (missing,) = value_errors(_LAW, {"bits": 4, "range": "fixed"}, "law")
        assert "law.range_value" in missing
        (stray,) = value_errors(
            _LAW, {"bits": 4, "range": "calibrated", "range_value": 1.0}, "law"
        )
        assert "law.range_value" in stray

    def test_closed_object_refuses_unknown_field(self):
        (message,) = value_errors(
            _LAW, {"bits": 4, "range": "calibrated", "wat": 1}, "law"
        )
        assert "law.wat" in message

    def test_open_object_tolerates_unknown_field(self):
        schema = ValueSchema(kind=FieldType.JSON, fields={"a": _INT_GE_2}, closed=False)
        assert value_errors(schema, {"a": 3, "other": 1}, "k") == []

    def test_non_mapping_object_value(self):
        assert value_errors(_LAW, 7, "law")

    def test_map_entries_are_judged_one_by_one(self):
        entry = ValueSchema(
            kind=FieldType.JSON,
            fields={"value": ValueSchema(kind=ANY),
                    "note": ValueSchema(kind=FieldType.STR, min_length=1)},
            required=("value", "note"),
        )
        schema = ValueSchema(kind=FieldType.JSON, entries=entry)
        assert value_errors(schema, {"x": {"value": 3, "note": "why"}}, "o") == []
        (message,) = value_errors(schema, {"x": {"value": 3}}, "o")
        assert "o.x.note" in message

    def test_core_type_list_judges_every_item(self):
        schema = ValueSchema(kind=FieldType.CORES, item=ValueSchema(
            kind=FieldType.JSON, fields={"count": ValueSchema(kind=FieldType.INT, low=1)},
            closed=False,
        ))
        assert value_errors(schema, [{"count": 1}, {"count": 2}], "cores") == []
        (message,) = value_errors(schema, [{"count": 1}, {"count": 0}], "cores")
        assert message.startswith("cores[1].count=0")
        assert value_errors(schema, {"count": 1}, "cores")

    def test_a_list_judges_its_length_and_every_item(self):
        """``min_length`` bounds a list as it bounds a string, in the same loop, and every item is judged by
        the item schema with its index in the path."""
        schema = ValueSchema(kind=FieldType.INT_LIST, item=ValueSchema(kind=FieldType.INT, low=1),
                             min_length=1)
        assert value_errors(schema, [1, 4], "k") == []
        assert value_errors(schema, [], "k") == ["k must be a list of at least 1 item(s)"]
        assert value_errors(schema, [0, 1.5, True], "k") == [
            "k[0]=0 is below the minimum 1", "k[1] must be int, got float (1.5)",
            "k[2] must be int, got bool (True)"]
        assert value_errors(schema, "1", "k") == ["k must be a list, got str ('1')"]
        assert value_errors(ValueSchema(kind=FieldType.INT_LIST, item=ValueSchema(kind=FieldType.INT)),
                            [], "k") == []
        assert describe(schema) == "list of INT >= 1 (non-empty)"

    def test_nested_schemas_walk_every_field(self):
        paths = [path for path, _ in nested_schemas(_LAW, "law")]
        assert paths == ["law.bits", "law.range", "law.range_value"]


class TestRegistryEntries:
    def test_scalar_schema_comes_from_type_bounds_and_options(self):
        schema = schema_of(REGISTRY["weight_bits"])
        assert schema is not None
        assert (schema.kind, schema.low, schema.high) == (FieldType.INT, 1, 32)
        assert schema_of(REGISTRY["model_config"]) is None  # a non-scalar key without a declared value schema
        # Review round 2: a recipe is judged by the fields its consumer reads (``RECIPE_SCHEMA``).
        assert schema_of(REGISTRY["training_recipe"]) is schema_of(REGISTRY["tuning_recipe"]) is RECIPE_SCHEMA

    def test_existing_scalar_messages_are_unchanged(self):
        assert entry_value_errors(REGISTRY["nf_scm_parity_samples"], -1) == [
            "nf_scm_parity_samples=-1 is below the minimum 0"
        ]
        assert entry_value_errors(REGISTRY["max_simulation_samples"], "lots") == [
            "max_simulation_samples must be int, got str ('lots')"
        ]

    def test_validator_rows_carry_the_loop_messages(self):
        rows = validate_against_registry({"readout_law": {"adc_bits": 1, "range": "calibrated",
                                                          "mult_bits": 8}})
        assert [r["key"] for r in rows] == ["readout_law"]
        assert rows[0]["message"].startswith("readout_law.adc_bits=1")
        assert rows[0]["rule_id"] == "field_domain"

    def test_judge_refuses_by_name_and_returns_the_value(self):
        entry = REGISTRY["inference_batch_K"]
        assert judge_declaration(entry, 16, run_domain="event") == 16
        with pytest.raises(DeclarationValueError, match="inference_batch_K"):
            judge_declaration(entry, 0, run_domain="event")
        assert issubclass(DeclarationValueError, ToolchainRefusal)
        assert issubclass(DeclarationValueError, ValueError)

    def test_domain_check_is_by_tag(self):
        value_key = REGISTRY["readout_law"]
        assert domain_errors(value_key, run_domain="value") == []
        (message,) = domain_errors(value_key, run_domain="event")
        assert message.startswith("readout_law")
        event_key = REGISTRY["simulation_steps"]
        assert domain_errors(event_key, run_domain="event") == []
        assert domain_errors(event_key, run_domain="value")
        assert domain_errors(REGISTRY["weight_bits"], run_domain="value") == []

    def test_describe_states_the_legal_values(self):
        assert describe(_INT_GE_2) == "INT >= 2"
        assert describe(_FLOAT_GT_0) == "FLOAT > 0"
        assert describe(_FRACTION) == "FLOAT in [0, 1)"
        assert describe(ValueSchema(kind=FieldType.ENUM, options=("a", "b"))) == "ENUM a | b"
        assert describe(ValueSchema(kind=FieldType.ENUM, options=("a",), reserved=("b", "c"))) == (
            "ENUM a (reserved: b, c)")


def _legal_a_reserved_b():
    return ValueSchema(kind=FieldType.ENUM, options=("a",), reserved=("b",))


def _closed_key(options, value_schema):
    return ConfigKeySchema(flat_key="k", group="run", owner="test", type=FieldType.ENUM,
                           category=Category.ADVANCED, label="K", doc="a closed key",
                           options=options, value_schema=value_schema)


class TestKeySchemaChecks:
    """The build-time checks of a key's value schema: the schema is of the key's own type, and the key's
    options (what the wizard offers) name the schema's legal values, then its reserved ones (what both
    doors judge by), in that order, so the two never disagree."""

    def test_options_naming_the_legal_then_the_reserved_values_build(self):
        assert _closed_key(("a", "b"), _legal_a_reserved_b()).resolved_options() == ("a", "b")

    @pytest.mark.parametrize("options", [("a",), ("b", "a"), ("a", "b", "c"), ("a", "c")],
                             ids=["reserved-missing", "reserved-first", "extra", "other"])
    def test_options_that_differ_from_the_schema_refuse(self, options):
        with pytest.raises(ValueError, match="legal, then reserved"):
            _closed_key(options, _legal_a_reserved_b())

    @pytest.mark.parametrize("schema", [ValueSchema(kind=FieldType.JSON),
                                        ValueSchema(kind=FieldType.STR, options=("a",))],
                             ids=["json", "str-with-the-same-options"])
    def test_a_value_schema_of_another_type_refuses(self, schema):
        with pytest.raises(ValueError, match="own type"):
            _closed_key(("a",), schema)

    def test_a_consumer_reading_rides_a_key_with_the_value_schema_both_doors_judge(self):
        """Both doors read a declaration through its consumer before the one judging loop, which judges
        a key by its value schema: a reading on a key without one refuses at build."""
        def reading(config):
            return config["k"]
        built = dataclasses.replace(_closed_key(("a", "b"), _legal_a_reserved_b()), consumer_reading=reading)
        assert built.consumer_reading is reading
        with pytest.raises(ValueError, match="consumer reading needs the value schema"):
            dataclasses.replace(_closed_key(("a",), None), consumer_reading=reading)

    @pytest.mark.parametrize("key", ["s_allocation", "nevresim_connectivity_mode"])
    def test_the_injected_consumer_legal_sets_satisfy_both_checks(self, key):
        entry = REGISTRY[key]
        assert entry.value_schema is not None and entry.value_schema.kind == entry.type.value
        assert entry.resolved_options() == entry.value_schema.options + entry.value_schema.reserved
        assert dataclasses.replace(entry) == entry
        with pytest.raises(ValueError, match="legal, then reserved"):
            dataclasses.replace(entry, options=tuple(reversed(entry.resolved_options())))


def test_the_loop_has_no_branch_on_a_key_name():
    """No registry key name appears as a string constant in the value-schema module."""
    tree = ast.parse(inspect.getsource(vs))
    constants = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    nested = {
        path.split(".")[-1]
        for entry in REGISTRY.values()
        if entry.value_schema is not None
        for path, _ in nested_schemas(entry.value_schema, entry.flat_key)
    }
    named = (set(REGISTRY) | nested) & constants
    assert not named, f"value_schema.py branches on key names: {sorted(named)}"
