"""Unit tests: semantic group mapping for pipeline steps.

Verifies that every step produced by get_pipeline_step_specs receives a
non-empty semantic group id via get_pipeline_semantic_group_by_step_name,
and that specific well-known steps map to expected group ids.
"""

import pytest

from mimarsinan.pipelining.pipelines.deployment_pipeline import (
    get_pipeline_step_specs,
    get_pipeline_semantic_group_by_step_name,
    _SEMANTIC_GROUP_BY_STEP_CLASS,
)


# ── Helper ───────────────────────────────────────────────────────────────────

def _base_config(**overrides) -> dict:
    return {
        "configuration_mode": "user",
        "spiking_mode": "rate",
        "activation_quantization": False,
        "weight_quantization": False,
        "pruning": False,
        "pruning_fraction": 0.0,
        "model_type": "mlp_mixer",
        **overrides,
    }


# ── Coverage: every step class must be in the group map ──────────────────────

class TestSemanticGroupMapCoverage:
    """_SEMANTIC_GROUP_BY_STEP_CLASS must cover every class returned by get_pipeline_step_specs."""

    @pytest.mark.parametrize("config_overrides,label", [
        ({}, "rate / no quant"),
        ({"spiking_mode": "ttfs", "activation_quantization": True, "weight_quantization": True}, "ttfs / full quant"),
        ({"model_type": "torch_custom", "weight_quantization": True}, "torch / weight quant"),
        ({"pruning": True, "pruning_fraction": 0.3}, "pruning"),
        ({"configuration_mode": "nas"}, "nas"),
        ({"weight_source": "pretrained.pt"}, "weight preloading"),
    ])
    def test_every_step_has_a_group(self, config_overrides, label):
        config = _base_config(**config_overrides)
        specs = get_pipeline_step_specs(config)
        for name, cls in specs:
            assert cls in _SEMANTIC_GROUP_BY_STEP_CLASS, (
                f"[{label}] Step class {cls.__name__!r} (step {name!r}) "
                f"has no entry in _SEMANTIC_GROUP_BY_STEP_CLASS."
            )

    def test_get_pipeline_semantic_group_returns_non_empty_for_all_steps(self):
        config = _base_config(
            spiking_mode="ttfs",
            activation_quantization=True,
            weight_quantization=True,
        )
        groups = get_pipeline_semantic_group_by_step_name(config)
        for name, g in groups.items():
            assert g and g != "other", (
                f"Step {name!r} resolved to group {g!r} — expected a real group id."
            )


# ── Spot-check: known steps → expected groups ────────────────────────────────

class TestKnownStepGroupMappings:

    def test_model_configuration_is_configuration_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config())
        assert groups["Model Configuration"] == "configuration"

    def test_model_building_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config())
        assert groups["Model Building"] == "model_building"

    def test_pretraining_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config())
        assert groups["Pretraining"] == "pretraining"

    def test_weight_preloading_same_group_as_pretraining(self):
        groups = get_pipeline_semantic_group_by_step_name(
            _base_config(weight_source="weights.pt")
        )
        assert groups["Weight Preloading"] == "pretraining"

    def test_activation_analysis_and_adaptation_share_activation_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config())
        assert groups["Activation Analysis"] == "activation"
        assert groups["Activation Adaptation"] == "activation"

    def test_clamp_adaptation_is_activation_group(self):
        groups = get_pipeline_semantic_group_by_step_name(
            _base_config(spiking_mode="ttfs", activation_quantization=False)
        )
        assert groups["Clamp Adaptation"] == "activation"

    def test_activation_quantization_steps_group(self):
        groups = get_pipeline_semantic_group_by_step_name(
            _base_config(activation_quantization=True)
        )
        assert groups["Activation Shifting"] == "activation_quantization"
        assert groups["Activation Quantization"] == "activation_quantization"

    def test_weight_quantization_steps_group(self):
        groups = get_pipeline_semantic_group_by_step_name(
            _base_config(weight_quantization=True)
        )
        assert groups["Weight Quantization"] == "weight_quantization"
        assert groups["Quantization Verification"] == "weight_quantization"

    def test_normalization_fusion_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config())
        assert groups["Normalization Fusion"] == "normalization"

    def test_soft_core_mapping_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config())
        assert groups["Soft Core Mapping"] == "soft_mapping"

    def test_core_quantization_verification_group(self):
        groups = get_pipeline_semantic_group_by_step_name(
            _base_config(weight_quantization=True)
        )
        assert groups["Core Quantization Verification"] == "core_verification"

    def test_coreflow_tuning_only_in_rate_mode(self):
        rate_groups = get_pipeline_semantic_group_by_step_name(_base_config(spiking_mode="rate"))
        assert "CoreFlow Tuning" in rate_groups
        assert rate_groups["CoreFlow Tuning"] == "coreflow_tuning"
        ttfs_groups = get_pipeline_semantic_group_by_step_name(
            _base_config(spiking_mode="ttfs")
        )
        assert "CoreFlow Tuning" not in ttfs_groups

    def test_hard_core_mapping_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config())
        assert groups["Hard Core Mapping"] == "hardware"

    def test_simulation_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config())
        assert groups["Simulation"] == "simulation"

    def test_architecture_search_is_configuration_group(self):
        groups = get_pipeline_semantic_group_by_step_name(_base_config(configuration_mode="nas"))
        assert groups["Architecture Search"] == "configuration"

    def test_torch_mapping_group(self):
        groups = get_pipeline_semantic_group_by_step_name(
            _base_config(model_type="torch_custom")
        )
        assert groups["Torch Mapping"] == "torch_mapping"

    def test_pruning_adaptation_group(self):
        groups = get_pipeline_semantic_group_by_step_name(
            _base_config(pruning=True, pruning_fraction=0.3)
        )
        assert groups["Pruning Adaptation"] == "pruning"


# ── API shape: groups and steps must stay in sync ────────────────────────────

class TestGroupsMatchStepOrder:

    def test_groups_dict_covers_exactly_the_composed_steps(self):
        """get_pipeline_semantic_group_by_step_name keys == step names in get_pipeline_step_specs."""
        config = _base_config(
            spiking_mode="ttfs",
            activation_quantization=True,
            weight_quantization=True,
        )
        specs = get_pipeline_step_specs(config)
        step_names = [n for n, _ in specs]
        groups = get_pipeline_semantic_group_by_step_name(config)
        assert list(groups.keys()) == step_names

    def test_group_count_matches_step_count(self):
        config = _base_config()
        specs = get_pipeline_step_specs(config)
        groups = get_pipeline_semantic_group_by_step_name(config)
        assert len(groups) == len(specs)
