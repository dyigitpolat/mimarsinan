"""The pretrained-weights regime in the config-schema layer (round-7).

The scalar ``pretrained_weight_source`` is retired. The injection contract is the
LIST ``pretrained_weight_sets`` (hidden, non-declarable), and the user's two
declarable knobs are ``preload_weights`` (the switch) and ``pretrained_weight_set``
(the choice among registered sets). Both are legality-bearing so the
legal-value-set law drives the switch's disabled/locked state and the selector's
options — no per-key JS.
"""

import pytest

from mimarsinan.config_schema.deployment_derivation import (
    derive_pipeline_runtime_parameters,
    legal_values_for,
    legality_bearing_keys,
)
from mimarsinan.config_schema.registry import Category, FieldType, REGISTRY
from mimarsinan.config_schema.validation import (
    legality_errors,
    validate_deployment_config,
)


def _sets(*records):
    return list(records)


_V1 = {
    "id": "imagenet1k_v1", "label": "V1", "task": "t", "dataset": "ImageNet-1K",
    "input_shape": [3, 224, 224], "num_classes": 1000, "source": "torchvision",
    "adapts_input_shape": True, "adapts_num_classes": True,
}
_V2 = {**_V1, "id": "imagenet1k_v2", "label": "V2", "source": "https://x/y.pt"}


class TestRetirement:
    def test_scalar_source_key_is_gone(self):
        assert "pretrained_weight_source" not in REGISTRY

    def test_the_injection_key_is_the_list(self):
        entry = REGISTRY["pretrained_weight_sets"]
        assert entry.category is Category.DERIVED
        assert entry.declarable is False
        assert entry.hidden is True
        assert entry.type is FieldType.JSON
        assert entry.provenance == "builder profile"

    def test_pretrained_weight_sets_is_not_authorable(self):
        errors = validate_deployment_config(
            {"deployment_parameters": {"pretrained_weight_sets": _sets(_V1)}}
        )
        assert any("pretrained_weight_sets" in e and "not declarable" in e
                   for e in errors)


class TestRegistryEntries:
    def test_preload_weights_is_the_basic_switch(self):
        entry = REGISTRY["preload_weights"]
        assert entry.category is Category.BASIC
        assert entry.type is FieldType.BOOL
        assert entry.group == "training"
        assert entry.legal_values is not None

    def test_pretrained_weight_set_is_derived_but_declarable(self):
        entry = REGISTRY["pretrained_weight_set"]
        assert entry.category is Category.DERIVED
        assert entry.declarable is True  # the user chooses among registered sets
        assert entry.group == "training"
        assert entry.provenance == "builder profile"
        assert entry.legal_values is not None
        assert entry.derived_default is not None

    def test_both_regime_knobs_are_legality_bearing(self):
        keys = set(legality_bearing_keys())
        assert {"preload_weights", "pretrained_weight_set"} <= keys


class TestPreloadLegalValueSet:
    def test_no_registration_is_not_judged(self):
        # None = legality does not apply (builder not consulted), distinct from
        # an empty set. The switch renders as an ordinary toggle.
        assert legal_values_for("preload_weights", {}) is None

    def test_empty_registration_locks_the_switch_off(self):
        cfg = {"pretrained_weight_sets": []}
        assert legal_values_for("preload_weights", cfg) == (False,)

    def test_registration_admits_both(self):
        cfg = {"pretrained_weight_sets": _sets(_V1)}
        assert legal_values_for("preload_weights", cfg) == (False, True)

    def test_explicit_source_locks_the_switch_on(self):
        cfg = {"pretrained_weight_sets": [], "weight_source": "/ckpt.pt"}
        assert legal_values_for("preload_weights", cfg) == (True,)


class TestWeightSetLegalValueSet:
    def test_single_registration_locks_the_choice(self):
        cfg = {"pretrained_weight_sets": _sets(_V1)}
        assert legal_values_for("pretrained_weight_set", cfg) == ("imagenet1k_v1",)

    def test_multiple_registrations_offer_a_choice(self):
        cfg = {"pretrained_weight_sets": _sets(_V1, _V2)}
        assert legal_values_for("pretrained_weight_set", cfg) == (
            "imagenet1k_v1", "imagenet1k_v2",
        )

    def test_model_config_applicability_filters_the_choice(self):
        gelu = {**_V1, "id": "gelu", "model_config_requires": {"base_activation": "GELU"}}
        cfg = {"pretrained_weight_sets": _sets(gelu),
               "model_config": {"base_activation": "ReLU"}}
        assert legal_values_for("pretrained_weight_set", cfg) == ()


class TestDeclaredValuesAreJudged:
    def test_preload_on_a_builder_that_registers_nothing_is_a_keyed_error(self):
        cfg = {"pretrained_weight_sets": [], "preload_weights": True}
        rows = legality_errors(cfg, cfg)
        keys = {r["key"] for r in rows}
        assert "preload_weights" in keys
        row = next(r for r in rows if r["key"] == "preload_weights")
        assert row["rule_id"] == "legal_value_set"
        # The remedy the RULE prescribes: clear the switch (accept OFF).
        assert any(rem["action"] == "set" and rem["value"] is False
                   for rem in row["remedies"])

    def test_an_unregistered_weight_set_id_is_a_keyed_error(self):
        cfg = {"pretrained_weight_sets": _sets(_V1), "preload_weights": True,
               "pretrained_weight_set": "imagenet1k_v99"}
        rows = legality_errors(cfg, cfg)
        row = next(r for r in rows if r["key"] == "pretrained_weight_set")
        assert row["rule_id"] == "legal_value_set"
        assert "imagenet1k_v99" in row["message"]

    def test_a_legal_selection_raises_nothing(self):
        cfg = {"pretrained_weight_sets": _sets(_V1, _V2), "preload_weights": True,
               "pretrained_weight_set": "imagenet1k_v2"}
        assert legality_errors(cfg, cfg) == []


class TestDerivationNeverRaisesOnAbsence:
    def test_absent_regime_leaves_the_choice_absent(self):
        dp = {"pretrained_weight_sets": _sets(_V1)}
        derive_pipeline_runtime_parameters(dp)
        # No preload → no selection; the key must not be forced to a value.
        assert dp.get("pretrained_weight_set") is None

    def test_the_derivation_fills_the_default_set_under_the_regime(self):
        dp = {"pretrained_weight_sets": _sets(_V1, _V2), "preload_weights": True}
        derive_pipeline_runtime_parameters(dp)
        assert dp["pretrained_weight_set"] == "imagenet1k_v1"

    def test_a_declared_illegal_choice_raises_for_programmatic_callers(self):
        dp = {"pretrained_weight_sets": _sets(_V1), "preload_weights": True,
              "pretrained_weight_set": "nope"}
        with pytest.raises(ValueError, match="not legal here"):
            derive_pipeline_runtime_parameters(dp)
