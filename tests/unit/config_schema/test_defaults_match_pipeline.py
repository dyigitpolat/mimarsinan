"""Ensure DeploymentPipeline uses config_schema defaults (no drift)."""

from mimarsinan.config_schema import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
    PIPELINE_MODE_PRESETS,
    apply_preset,
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline


class TestDefaultsMatchPipeline:
    def test_pipeline_class_defaults_are_schema_copies(self):
        assert DeploymentPipeline.default_deployment_parameters is DEFAULT_DEPLOYMENT_PARAMETERS
        assert DeploymentPipeline.default_platform_constraints is DEFAULT_PLATFORM_CONSTRAINTS

    def test_apply_preset_delegates_to_schema(self):
        params = {}
        DeploymentPipeline.apply_preset("phased", params)
        expected = {}
        apply_preset("phased", expected)
        assert params == expected

    def test_sanafe_keys_in_defaults(self):
        d = get_default_deployment_parameters()
        assert "enable_sanafe_simulation" in d
        assert "sanafe_arch_preset" in d

    def test_platform_defaults_include_scheduling_keys(self):
        p = get_default_platform_constraints()
        assert "allow_coalescing" in p
        assert "max_schedule_passes" in p


class TestConfigKeysInventory:
    def test_every_defaulted_key_is_a_known_key(self):
        """CONFIG_KEYS_SET must cover DEFAULT_DEPLOYMENT_PARAMETERS (no drift)."""
        from mimarsinan.config_schema.defaults import (
            CONFIG_KEYS_SET,
            DEFAULT_DEPLOYMENT_PARAMETERS,
        )

        assert set(DEFAULT_DEPLOYMENT_PARAMETERS) <= CONFIG_KEYS_SET
