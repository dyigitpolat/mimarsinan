"""THE representability test: every tier config round-trips through the wizard schema.

The wizard renders entirely from the config-key registry, so Python-side
round-trip through parse/emit IS UI representability: every tier-0/0.1/1/2
config must be buildable from scratch and fully viewable/editable as a
template. This pins the wizard as the configurability SSOT.
"""

import glob
import json
import os

import pytest

from mimarsinan.config_schema.registry import (
    Category,
    FieldType,
    REGISTRY,
    parse_deployment_document,
)
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.gui.wizard.emit import emit_deployment_config

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_PATHS = sorted(
    glob.glob(os.path.join(_REPO_ROOT, "test_configs", "tier*", "t*.json"))
)


def _config_id(path: str) -> str:
    return os.path.relpath(path, os.path.join(_REPO_ROOT, "test_configs"))


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_the_tier_matrix_is_present() -> None:
    assert len(_CONFIG_PATHS) >= 48, "tier configs missing; representability undefined"


@pytest.mark.parametrize("path", _CONFIG_PATHS, ids=_config_id)
class TestRepresentability:
    def test_parse_every_key_is_schema_known(self, path):
        """No unknown keys: this alone would have caught endpoint_floor_wall_s."""
        parsed = parse_deployment_document(_load(path))
        assert parsed.unknown == []

    def test_categorize_pure_user_intent(self, path):
        """Tier configs never declare runtime or non-declarable derived keys."""
        parsed = parse_deployment_document(_load(path))
        for key in parsed.known_flat_keys():
            entry = REGISTRY[key]
            assert entry.category is not Category.RUNTIME, key
            if entry.category is Category.DERIVED:
                assert entry.declarable, (
                    f"{key} is derivation-owned; tier configs must not pin it"
                )

    def test_roundtrip_emit_is_lossless(self, path):
        """emit(parse(config)) == config modulo key order — the Deploy builder
        (the same one the wizard calls) must not drop or invent a single key."""
        config = _load(path)
        emitted = emit_deployment_config(config)
        assert json.dumps(emitted, sort_keys=True) == json.dumps(config, sort_keys=True)

    def test_validate_resolves_with_zero_errors(self, path):
        resolution = resolve_draft(_load(path))
        assert resolution.errors == []
        assert resolution.unknown_keys == []

    def test_widgetable_every_value_renders(self, path):
        """Enum values sit in the schema options, numerics inside bounds, types
        match — a 'Novena missing from firing modes' class of bug fails HERE,
        in Python, not in a browser."""
        parsed = parse_deployment_document(_load(path))
        for key, value in parsed.known_flat_keys().items():
            entry = REGISTRY[key]
            if value is None:
                continue
            if entry.type is FieldType.ENUM:
                options = entry.resolved_options() or ()
                assert value in options, f"{key}={value!r} not in {options}"
            elif entry.type is FieldType.BOOL:
                assert isinstance(value, bool), key
            elif entry.type in (FieldType.INT, FieldType.FLOAT):
                assert isinstance(value, (int, float)) and not isinstance(value, bool), key
                if entry.bounds is not None:
                    lo, hi = entry.bounds
                    assert lo is None or value >= lo, f"{key}={value} < {lo}"
                    assert hi is None or value <= hi, f"{key}={value} > {hi}"

    def test_model_config_fields_match_the_builder_schema(self, path):
        """model_config is opaque to the registry; its fields must match the
        per-builder schema the wizard renders from."""
        config = _load(path)
        dp = config.get("deployment_parameters") or {}
        model_type = dp.get("model_type")
        model_config = dp.get("model_config") or {}
        if not model_type or not model_config:
            return
        from mimarsinan.pipelining.core.registry.model_registry import (
            get_model_config_schema,
        )
        schema_keys = {field["key"] for field in get_model_config_schema(model_type)}
        unknown = set(model_config) - schema_keys
        assert not unknown, (
            f"{model_type} model_config fields not in the builder schema: {unknown}"
        )


class TestRepresentabilityCanary:
    def test_the_historical_regression_is_caught(self):
        """A config carrying the retired endpoint_floor_wall_s key must be
        flagged unknown (it was silently dropped for months) and still
        survive emission verbatim."""
        config = {
            "experiment_name": "canary",
            "deployment_parameters": {"endpoint_floor_wall_s": 60},
        }
        parsed = parse_deployment_document(config)
        assert parsed.unknown == ["deployment_parameters.endpoint_floor_wall_s"]
        emitted = emit_deployment_config(config)
        assert emitted["deployment_parameters"]["endpoint_floor_wall_s"] == 60
