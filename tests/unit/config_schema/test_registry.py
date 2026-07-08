"""The config-key registry: coverage vs the live CONFIG_KEYS_SET, docs, relevance."""

import json

import pytest

from mimarsinan.config_schema.defaults import (
    CONFIG_KEYS_SET,
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
)
from mimarsinan.config_schema.namespaced_schema import KEY_SPECS
from mimarsinan.config_schema.registry import (
    Category,
    ConfigKeySchema,
    FieldType,
    NON_PIPELINE_DOC_KEYS,
    REGISTRY,
    Relevance,
    parse_deployment_document,
    serialize_registry,
)


class TestCoverage:
    """The registry is validated against the live key inventory, never hand-trusted."""

    def test_registry_equals_config_keys_plus_document_extras(self):
        assert set(REGISTRY) == set(CONFIG_KEYS_SET) | set(NON_PIPELINE_DOC_KEYS)

    def test_flagship_orphans_are_registered(self):
        # The keys whose non-registration caused silent template loss (S4).
        for key in ("endpoint_floor_steps", "tuning_batch_size", "conversion_draws",
                    "wq_endpoint_recovery_steps", "nf_scm_parity_samples",
                    "capacity_gate", "pretrain_floor_chance_multiple"):
            assert key in REGISTRY, key

    def test_defaults_are_injected_from_the_defaults_ssot(self):
        for key, value in DEFAULT_DEPLOYMENT_PARAMETERS.items():
            assert REGISTRY[key].default == value, key
        for key, value in DEFAULT_PLATFORM_CONSTRAINTS.items():
            assert REGISTRY[key].default == value, key

    def test_keyspec_table_is_derived_from_the_registry(self):
        assert set(KEY_SPECS) == set(REGISTRY)
        for key, spec in KEY_SPECS.items():
            entry = REGISTRY[key]
            assert (spec.group, spec.owner, spec.derivation, spec.exposure) == (
                entry.group, entry.owner, entry.derivation, entry.exposure
            ), key


class TestEntryQuality:
    def test_every_entry_has_label_and_doc(self):
        for key, entry in REGISTRY.items():
            assert entry.label.strip(), key
            assert len(entry.doc.strip()) >= 15, key

    def test_enum_entries_resolve_options(self):
        for key, entry in REGISTRY.items():
            if entry.type is FieldType.ENUM:
                options = entry.resolved_options()
                assert options and len(options) >= 2, key

    def test_derived_entries_carry_inputs_and_why(self):
        for key, entry in REGISTRY.items():
            if entry.category is Category.DERIVED:
                assert entry.derived_from, key
                assert entry.why is not None, key

    def test_category_derivation_pairing_is_enforced(self):
        with pytest.raises(ValueError):
            ConfigKeySchema(
                flat_key="bogus", group="run", owner="x", type=FieldType.BOOL,
                category=Category.DERIVED, label="Bogus", doc="A bogus derived key.",
            )


class TestRelevance:
    def test_json_codec_round_trips_every_entry(self):
        samples = (
            {"spiking_mode": "lif", "pruning": True, "s_allocation": "explicit",
             "enable_sanafe_simulation": True, "weight_source": "vgg16",
             "model_config_mode": "user", "hw_config_mode": "search",
             "enable_loihi_simulation": True, "enable_nevresim_simulation": True,
             "allow_scheduling": True},
            {"spiking_mode": "ttfs_cycle_based", "pruning": False},
            {},
        )
        for key, entry in REGISTRY.items():
            rebuilt = Relevance.from_json(entry.relevant.to_json())
            for sample in samples:
                assert rebuilt.evaluate(sample) == entry.relevant.evaluate(sample), key

    def test_combinators(self):
        r = Relevance.all_of(
            Relevance.when("spiking_mode", in_=("lif",)),
            Relevance.when_true("pruning"),
        )
        assert r.evaluate({"spiking_mode": "lif", "pruning": True})
        assert not r.evaluate({"spiking_mode": "lif", "pruning": False})
        assert Relevance.when_set("weight_source").evaluate({"weight_source": "x"})
        assert not Relevance.when_set("weight_source").evaluate({"weight_source": None})


class TestTaxonomy:
    """Dataset-side facts live in 'workload'; architecture-side in 'model';
    the hardware-search switch with the hardware it searches. The wizard's
    step flow (Workload -> Model -> Deployment -> Tuning -> Review) maps
    whole groups to steps, so this split IS the step assignment."""

    def test_workload_group_is_dataset_side(self):
        for key in ("data_provider_name", "datasets_path", "preprocessing",
                    "input_data_scale", "spike_encoding_seed"):
            assert REGISTRY[key].group == "workload", key

    def test_model_group_is_architecture_side(self):
        for key in ("model_config_mode", "model_type", "model_config",
                    "model_factory", "arch_search", "pruning",
                    "pruning_fraction", "prune_sparsity", "weight_source",
                    "preload_weights", "pretrained_weight_source",
                    "clamp_cuda_assert_prone"):
            assert REGISTRY[key].group == "model", key

    def test_hw_search_switch_lives_with_hardware(self):
        assert REGISTRY["hw_config_mode"].group == "hardware"


class TestSerialization:
    def test_payload_is_json_safe_and_complete(self):
        payload = serialize_registry()
        json.dumps(payload)
        assert len(payload["groups"]) == 9
        assert set(payload["keys"]) == set(REGISTRY)
        record = payload["keys"]["endpoint_floor_steps"]
        assert record["unit"] == "steps"
        assert record["category"] == "advanced"

    def test_relevance_trees_are_serialized(self):
        payload = serialize_registry()
        tree = payload["keys"]["ttfs_cycle_schedule"]["relevant"]
        assert tree == {"op": "in", "key": "spiking_mode", "values": ["ttfs_cycle_based"]}


class TestParse:
    def test_unknown_keys_are_reported_never_dropped(self):
        parsed = parse_deployment_document({
            "experiment_name": "x",
            "deployment_parameters": {"endpoint_floor_wall_s": 60, "lr": 0.01},
            "platform_constraints": {"weight_bits": 5, "bogus_pc": 1},
            "mystery_top": True,
        })
        assert "deployment_parameters.endpoint_floor_wall_s" in parsed.unknown
        assert "platform_constraints.bogus_pc" in parsed.unknown
        assert "mystery_top" in parsed.unknown
        assert parsed.dp == {"lr": 0.01}
        assert parsed.pc == {"weight_bits": 5}

    def test_nested_core_and_recipe_fields_are_checked(self):
        parsed = parse_deployment_document({
            "platform_constraints": {
                "cores": [{"max_axons": 1, "max_neurons": 1, "count": 1, "wat": 2}],
            },
            "deployment_parameters": {
                "training_recipe": {"optimizer": "adamw", "unknown_recipe_field": 1},
            },
        })
        assert any("cores[0].wat" in path for path in parsed.unknown)
        assert any("unknown_recipe_field" in path for path in parsed.unknown)

    def test_wizard_hw_search_platform_shape_parses(self):
        parsed = parse_deployment_document({
            "platform_constraints": {
                "mode": "auto",
                "auto": {"fixed": {"weight_bits": 4}, "search_space": {"num_core_types": 2}},
            },
        })
        assert parsed.pc["weight_bits"] == 4
        assert parsed.pc["search_space"] == {"num_core_types": 2}
        assert parsed.unknown == []

    def test_meta_keys_pass_through(self):
        parsed = parse_deployment_document({"_continue_from_run_id": "run_1"})
        assert parsed.meta == {"_continue_from_run_id": "run_1"}
        assert parsed.unknown == []
