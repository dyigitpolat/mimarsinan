"""Scale Migration step gating and ordering pins.

The step is registry-gated (``scale_migration``, default OFF) and must sit
between Pruning Adaptation and Activation Analysis: it needs committed prune
masks (row/column scaling preserves zeros but changes magnitude ranking) and a
not-yet-installed theta (rescaling weights under a stale scalar theta is no
longer function-preserving through the clamp).
"""

import pytest

from mimarsinan.config_schema.registry import effective_value, schema_for
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
    get_pipeline_semantic_group_by_step_name,
    get_pipeline_step_specs,
)
from mimarsinan.pipelining.pipeline_steps.adaptation.scale_migration_step import (
    ScaleMigrationStep,
)
from mimarsinan.transformations.channel_scale_equalization import DEFAULT_CLIP_RATIO


def _config(**overrides) -> dict:
    cfg = {
        "configuration_mode": "user",
        "spiking_mode": "lif",
        "activation_quantization": False,
        "weight_quantization": False,
        "model_type": "mlp_mixer",
    }
    cfg.update(overrides)
    return cfg


def _step_names(config: dict) -> list[str]:
    return [name for name, _ in get_pipeline_step_specs(config)]


class TestFlagOffByteIdentical:
    def test_default_off_in_every_mode(self):
        for spiking in ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            names = _step_names(_config(spiking_mode=spiking))
            assert "Scale Migration" not in names

    def test_absent_flag_matches_explicit_false(self):
        for spiking in ("lif", "ttfs_quantized"):
            base = _step_names(_config(spiking_mode=spiking))
            explicit = _step_names(
                _config(spiking_mode=spiking, scale_migration=False)
            )
            assert base == explicit

    def test_applies_to_follows_the_plan_gate(self):
        assert not ScaleMigrationStep.applies_to(DeploymentPlan.resolve(_config()))
        assert ScaleMigrationStep.applies_to(
            DeploymentPlan.resolve(_config(scale_migration=True))
        )


class TestOrderingPin:
    def test_between_pruning_adaptation_and_activation_analysis(self):
        names = _step_names(
            _config(scale_migration=True, pruning=True, pruning_fraction=0.3)
        )
        assert "Scale Migration" in names
        idx = names.index("Scale Migration")
        assert idx == names.index("Activation Analysis") - 1
        assert idx > names.index("Pruning Adaptation")

    def test_immediately_before_activation_analysis_without_pruning(self):
        names = _step_names(_config(scale_migration=True))
        assert "Pruning Adaptation" not in names
        idx = names.index("Scale Migration")
        assert idx == names.index("Activation Analysis") - 1
        assert idx > names.index("Model Building")

    def test_step_carries_a_semantic_group(self):
        groups = get_pipeline_semantic_group_by_step_name(
            _config(scale_migration=True)
        )
        assert groups["Scale Migration"] == "activation"


class TestConfigResolution:
    def test_plan_defaults(self):
        plan = DeploymentPlan.resolve(_config())
        assert plan.scale_migration_enabled is False
        assert plan.scale_migration_clip_ratio == pytest.approx(DEFAULT_CLIP_RATIO)

    def test_plan_explicit_values(self):
        plan = DeploymentPlan.resolve(
            _config(scale_migration=True, scale_migration_clip_ratio=2.5)
        )
        assert plan.scale_migration_enabled is True
        assert plan.scale_migration_clip_ratio == pytest.approx(2.5)

    def test_registry_declares_the_knobs(self):
        assert schema_for("scale_migration") is not None
        assert schema_for("scale_migration_clip_ratio") is not None
        assert effective_value({}, "scale_migration") is False
        assert effective_value({}, "scale_migration_clip_ratio") == pytest.approx(
            DEFAULT_CLIP_RATIO
        )
