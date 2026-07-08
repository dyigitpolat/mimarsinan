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
    deployment-side conversion knobs in 'conversion'; the search concern in
    'co_search' (it co-optimizes BOTH model and hardware — nesting it under
    either card misleads). The wizard's workbench sections host whole
    groups, so this split IS the section assignment."""

    def test_workload_group_is_dataset_side(self):
        for key in ("data_provider_name", "datasets_path", "preprocessing",
                    "input_data_scale", "spike_encoding_seed", "num_workers"):
            assert REGISTRY[key].group == "workload", key

    def test_data_loading_settings_render_primary_in_workload(self):
        """Round-3 defect 1: data-loading settings are NOT advanced — the
        preprocessing spec and the datasets path are primary Workload fields."""
        for key in ("data_provider_name", "preprocessing", "datasets_path"):
            assert REGISTRY[key].category is Category.BASIC, key

    def test_model_group_is_architecture_side(self):
        for key in ("model_config_mode", "model_type", "model_config",
                    "model_factory", "clamp_cuda_assert_prone"):
            assert REGISTRY[key].group == "model", key

    def test_pretrained_model_config_lives_in_training(self):
        """Round-3 defect 4: the pretrained-weight regime configures the
        TRAINING path (fine-tune instead of from-scratch), so its keys live
        in the training group with the weight source primary."""
        for key in ("weight_source", "preload_weights", "pretrained_weight_source"):
            assert REGISTRY[key].group == "training", key
        assert REGISTRY["weight_source"].category is Category.BASIC
        for key in ("finetune_epochs", "finetune_lr"):
            entry = REGISTRY[key]
            assert entry.group == "training", key
            # Fine-tune knobs are the point of the pretrained regime: primary
            # exactly while a weight source is declared.
            assert entry.promote_when is not None, key
            assert entry.promote_when.evaluate({"weight_source": "torchvision"}), key
            assert not entry.promote_when.evaluate({"weight_source": None}), key

    def test_mapping_strategy_hosts_the_mapping_choices(self):
        """Round-3 defect 5 (amended): capability = what the hardware CAN do
        (Hardware card); strategy = what we CHOOSE when mapping (the
        Mapping-strategy panel in Co-Design): scheduling, encoding placement,
        the pruning family."""
        for key in ("allow_scheduling", "max_schedule_passes",
                    "scheduling_latency_weight", "encoding_layer_placement",
                    "pruning", "pruning_fraction", "prune_sparsity"):
            assert REGISTRY[key].group == "mapping_strategy", key

    def test_capabilities_stay_on_the_hardware_card(self):
        for key in ("allow_coalescing", "allow_neuron_splitting",
                    "allow_per_layer_s", "has_bias", "cores"):
            assert REGISTRY[key].group == "hardware", key

    def test_tuning_recipe_lives_with_the_tuning_controller(self):
        """Round-3 defect 7: the tuning recipe configures the adaptation
        tuners, not pretraining — it belongs to the tuning group."""
        assert REGISTRY["tuning_recipe"].group == "tuning"
        assert REGISTRY["training_recipe"].group == "training"

    def test_pruning_is_a_mapping_strategy_never_an_architecture_property(self):
        """Pruning is something we CHOOSE when mapping (it shrinks the mapped
        weight surface), never an architecture property — it lives on the
        Co-Design Mapping-strategy panel (round-3 amendment)."""
        for key in ("pruning", "pruning_fraction", "prune_sparsity"):
            assert REGISTRY[key].group == "mapping_strategy", key

    def test_co_search_group_hosts_the_search_concern(self):
        """The search settings drive model AND hardware co-search, so they
        live in their own concern group, never under Model or Hardware."""
        for key in ("arch_search", "search_space"):
            assert REGISTRY[key].group == "co_search", key

    def test_config_mode_switches_live_with_the_concern_they_declare(self):
        """The mode selectors are per-concern provenance switches (hand vs
        search) — each stays on the card whose config source it declares;
        their 'search' position is what activates the co-search panel."""
        assert REGISTRY["model_config_mode"].group == "model"
        assert REGISTRY["hw_config_mode"].group == "hardware"

    def test_co_search_group_declares_its_off_state(self):
        group = next(g for g in serialize_registry()["groups"]
                     if g["id"] == "co_search")
        assert group["empty_state"]


class TestModeHonesty:
    """Keys a policy or derivation owns must never render as knobs."""

    def test_simulator_enables_are_derived_with_a_declarable_off(self):
        """Round-3 defect 6: a supported vehicle defaults ON per the mode
        recipe and the user may declare it OFF (a legitimate, stored
        override); an unsupported vehicle stays capability-off. The keys are
        DERIVED (the recipe owns the default) but DECLARABLE (user-off
        survives emission), and they expose machine-readable support meta."""
        for key in ("enable_nevresim_simulation", "enable_loihi_simulation",
                    "enable_sanafe_simulation"):
            entry = REGISTRY[key]
            assert entry.category is Category.DERIVED, key
            assert entry.declarable is True, key
            assert "spiking_mode" in entry.derived_from, key
            assert entry.why is not None, key
            assert entry.meta is not None, key
            assert entry.meta({"spiking_mode": "lif"}).keys() >= {"supported"}, key

    def test_cycle_accurate_lif_forward_is_recipe_owned_not_a_knob(self):
        """Round-3 defect 8 (defaults audit): the cycle-accurate LIF forward
        is what makes the QAT train-forward bit-exact to the deployed eval
        forward — a correctness mechanism the LIF recipe always folds ON.
        It must not render as a knob."""
        entry = REGISTRY["cycle_accurate_lif_forward"]
        assert entry.category is Category.DERIVED
        assert entry.declarable is False
        assert entry.derived_from == ("spiking_mode",)
        assert entry.why is not None

    def test_core_maxima_are_derived_from_the_core_grid(self):
        """max_axons/max_neurons are derivable from cores; they render as
        derived chips (documents may still declare a consistent value)."""
        for key in ("max_axons", "max_neurons"):
            entry = REGISTRY[key]
            assert entry.category is Category.DERIVED, key
            assert entry.declarable is True, key
            assert entry.derived_from == ("cores",), key

    def test_search_settings_are_promoted_in_search_mode(self):
        """A knob that is the point of the current mode is never 'advanced':
        promote_when holds exactly in the mode the knob configures."""
        arch = REGISTRY["arch_search"]
        assert arch.promote_when is not None
        assert arch.promote_when.evaluate({"model_config_mode": "search"})
        assert not arch.promote_when.evaluate(
            {"model_config_mode": "user", "hw_config_mode": "fixed"})
        space = REGISTRY["search_space"]
        assert space.promote_when is not None
        assert space.promote_when.evaluate({"hw_config_mode": "search"})
        assert not space.promote_when.evaluate({"hw_config_mode": "fixed"})

    def test_promote_when_requires_an_advanced_category(self):
        with pytest.raises(ValueError):
            ConfigKeySchema(
                flat_key="bogus", group="run", owner="x", type=FieldType.BOOL,
                category=Category.BASIC, label="Bogus", doc="A bogus basic key.",
                promote_when=Relevance.when_true("pruning"),
            )

    def test_search_owned_hand_fields_declare_their_provider(self):
        """When a search mode makes a hand field irrelevant, the card must
        say WHO owns it instead of silently dropping it: the keys the
        co-search discovers carry provided_by='co_search'."""
        for key in ("model_config", "cores"):
            entry = REGISTRY[key]
            assert entry.provided_by == "co_search", key
            assert entry.relevant.op != "always", key

    def test_model_type_is_consumed_in_every_search_mode(self):
        """ArchitectureSearchStep reads model_type unconditionally (it is
        the builder family whose space the search explores), so the key is
        always relevant — hiding it under search would be a lie."""
        entry = REGISTRY["model_type"]
        assert entry.relevant.evaluate({"model_config_mode": "search"})
        assert entry.relevant.evaluate({"model_config_mode": "user"})

    def test_cores_are_search_owned_under_hw_search(self):
        """The hardware co-search discovers the core grid, so the hand
        editor only exists while hw_config_mode='fixed'."""
        entry = REGISTRY["cores"]
        assert entry.relevant.evaluate({"hw_config_mode": "fixed"})
        assert not entry.relevant.evaluate({"hw_config_mode": "search"})

    def test_provided_by_requires_a_conditional_relevance(self):
        with pytest.raises(ValueError):
            ConfigKeySchema(
                flat_key="bogus", group="run", owner="x", type=FieldType.BOOL,
                category=Category.BASIC, label="Bogus", doc="A bogus basic key.",
                provided_by="co_search",
            )

    def test_no_default_user_keys_state_what_empty_means(self):
        """Every empty box must say what empty does; the flagship
        workload-profile keys carry an explicit empty_means."""
        for key in ("eval_subsample_target", "tuning_step_cap_epochs",
                    "prefix_stage_lr", "endpoint_floor_lr",
                    "proven_recovery_depth", "endpoint_floor_steps",
                    "wq_endpoint_recovery_steps", "conversion_draws",
                    "weight_source", "finetune_lr", "tuning_batch_size",
                    "start_step", "stop_step", "input_data_scale",
                    "pruning_fraction"):
            entry = REGISTRY[key]
            assert entry.empty_means, key


class TestSerialization:
    def test_payload_is_json_safe_and_complete(self):
        payload = serialize_registry()
        json.dumps(payload)
        assert len(payload["groups"]) == 11
        assert set(payload["keys"]) == set(REGISTRY)
        record = payload["keys"]["endpoint_floor_steps"]
        assert record["unit"] == "steps"
        assert record["category"] == "advanced"

    def test_provided_by_is_serialized(self):
        payload = serialize_registry()
        assert payload["keys"]["model_config"]["provided_by"] == "co_search"
        assert payload["keys"]["cores"]["provided_by"] == "co_search"
        assert payload["keys"]["lr"]["provided_by"] is None

    def test_relevance_trees_are_serialized(self):
        payload = serialize_registry()
        tree = payload["keys"]["ttfs_cycle_schedule"]["relevant"]
        assert tree == {"op": "in", "key": "spiking_mode", "values": ["ttfs_cycle_based"]}

    def test_promote_when_and_empty_means_are_serialized(self):
        payload = serialize_registry()
        arch = payload["keys"]["arch_search"]
        assert arch["promote_when"] is not None
        assert arch["promote_when"]["op"]
        assert payload["keys"]["lr"]["promote_when"] is None
        assert payload["keys"]["tuning_batch_size"]["empty_means"]


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
