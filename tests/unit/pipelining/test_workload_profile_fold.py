"""The registry hooks and pipeline fold of workload profiles (plan.workload)."""

from mimarsinan.common.workload_profile import (
    DataWorkloadProfile,
    ModelWorkloadProfile,
    ResolvedWorkloadProfile,
)
from mimarsinan.data_handling.data_provider import DataProvider
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
    apply_workload_profiles,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


class _ProfiledProvider(DataProvider):
    def __init__(self):
        super().__init__("<test>")

    def workload_profile(self):
        return DataWorkloadProfile(
            input_value_range=(0.0, 2.5), eval_subsample_target=10000
        )


class TestProviderHook:
    def test_base_provider_registers_the_all_none_profile(self):
        assert DataProvider("<test>").workload_profile() == DataWorkloadProfile()


class TestBuilderHook:
    def test_unknown_model_type_has_no_profile(self):
        assert ModelRegistry.get_workload_profile("no_such_model") is None

    def test_undeclared_builder_has_no_profile(self):
        assert ModelRegistry.get_workload_profile("simple_mlp") is None

    def test_declared_builder_profile_is_returned(self, monkeypatch):
        class _Builder:
            @classmethod
            def workload_profile(cls):
                return ModelWorkloadProfile(prefix_stage_lr=5e-4)

        monkeypatch.setitem(
            ModelRegistry._registry,
            "_test_profiled",
            {"label": "t", "category": "torch", "builder_cls": _Builder},
        )
        profile = ModelRegistry.get_workload_profile("_test_profiled")
        assert profile == ModelWorkloadProfile(prefix_stage_lr=5e-4)

    def test_builders_registry_is_the_model_registry_view(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        view = ModelRegistry.builder_classes()
        assert BUILDERS_REGISTRY == view
        assert len(view) >= 13


class TestPipelineFold:
    def test_registered_data_profile_lands_beneath_explicit_config(self):
        config = {"model_type": "simple_mlp", "eval_subsample_target": 640}
        apply_workload_profiles(config, _ProfiledProvider())
        assert config["input_data_scale"] == 2.5
        assert config["eval_subsample_target"] == 640

    def test_declared_model_profile_beats_the_data_profile(self, monkeypatch):
        class _Builder:
            @classmethod
            def workload_profile(cls):
                return ModelWorkloadProfile(pretrained_weight_source="hub")

        monkeypatch.setitem(
            ModelRegistry._registry,
            "_test_profiled",
            {"label": "t", "category": "torch", "builder_cls": _Builder},
        )
        config = {"model_type": "_test_profiled"}
        apply_workload_profiles(config, _ProfiledProvider())
        assert config["pretrained_weight_source"] == "hub"
        assert config["input_data_scale"] == 2.5

    def test_plan_carries_the_resolved_workload(self):
        plan = DeploymentPlan.resolve({"spiking_mode": "lif"})
        assert plan.workload == ResolvedWorkloadProfile()

    def test_plan_workload_reads_the_folded_keys(self):
        config = {"spiking_mode": "lif", "model_type": "simple_mlp"}
        apply_workload_profiles(config, _ProfiledProvider())
        plan = DeploymentPlan.resolve(config)
        assert plan.workload.input_data_scale == 2.5
        assert plan.workload.eval_subsample_target == 10000
