"""The registry hooks and pipeline fold of workload profiles (plan.workload)."""

import pytest

from mimarsinan.common.workload_profile import (
    DataWorkloadProfile,
    ModelWorkloadProfile,
    ResolvedWorkloadProfile,
)
from mimarsinan.data_handling.data_provider import DataProvider
from mimarsinan.pipelining.core.deployment_plan import (
    DeploymentPlan,
    resolve_weight_source,
)
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

    def test_undeclared_builder_has_the_empty_profile(self):
        """A KNOWN builder always has a registration; only an unknown model_type
        has none. `[]` ("registers no pretrained weights") is a claim, absence is not."""
        profile = ModelRegistry.get_workload_profile("simple_mlp")
        assert profile == ModelWorkloadProfile()
        assert profile.pretrained_weight_sets == ()

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
                return ModelWorkloadProfile(prefix_stage_lr=7e-4)

        monkeypatch.setitem(
            ModelRegistry._registry,
            "_test_profiled",
            {"label": "t", "category": "torch", "builder_cls": _Builder},
        )
        config = {"model_type": "_test_profiled"}
        apply_workload_profiles(config, _ProfiledProvider())
        assert config["prefix_stage_lr"] == 7e-4
        assert config["input_data_scale"] == 2.5

    def test_the_fold_always_states_what_the_builder_registers(self, monkeypatch):
        """A builder that declares no hook still injects the EMPTY registration,
        so `pretrained_weight_sets: []` is a claim and never an absence."""
        class _Bare:
            pass

        monkeypatch.setitem(
            ModelRegistry._registry,
            "_test_bare",
            {"label": "t", "category": "torch", "builder_cls": _Bare},
        )
        config = {"model_type": "_test_bare"}
        apply_workload_profiles(config, _ProfiledProvider())
        assert config["pretrained_weight_sets"] == []

    def test_an_unknown_model_type_injects_no_registration(self):
        config = {"model_type": "not_a_builder"}
        apply_workload_profiles(config, _ProfiledProvider())
        assert "pretrained_weight_sets" not in config

    def test_plan_carries_the_resolved_workload(self):
        plan = DeploymentPlan.resolve({"spiking_mode": "lif"})
        assert plan.workload == ResolvedWorkloadProfile()

    def test_plan_workload_reads_the_folded_keys(self):
        config = {"spiking_mode": "lif", "model_type": "simple_mlp"}
        apply_workload_profiles(config, _ProfiledProvider())
        plan = DeploymentPlan.resolve(config)
        assert plan.workload.input_data_scale == 2.5
        assert plan.workload.eval_subsample_target == 10000


class TestShippedBuilderRegistrations:
    def test_torchvision_builders_register_pretrained_weight_sets(self):
        for model_type in (
            "torch_vgg16", "torch_vit", "torch_squeezenet11", "torch_resnet50",
        ):
            profile = ModelRegistry.get_workload_profile(model_type)
            assert profile is not None, model_type
            assert profile.pretrained_weight_sets, model_type
            for weight_set in profile.pretrained_weight_sets:
                assert weight_set.source == "torchvision", model_type

    def test_vit_registers_the_clamp_cuda_assert_flag(self):
        profile = ModelRegistry.get_workload_profile("torch_vit")
        assert profile is not None
        assert profile.clamp_cuda_assert_prone is True

    def test_native_builders_register_the_empty_weight_set(self):
        profile = ModelRegistry.get_workload_profile("deep_cnn")
        assert profile is not None
        assert profile.pretrained_weight_sets == ()

    def test_unknown_model_type_has_no_registration(self):
        assert ModelRegistry.get_workload_profile("not_a_builder") is None


def _sets_config(**extra):
    """The config a run sees after the builder registration is folded in."""
    config = {
        "pretrained_weight_sets": [
            {"id": "imagenet1k_v1", "label": "V1", "task": "t", "dataset": "D",
             "input_shape": [3, 224, 224], "num_classes": 1000,
             "source": "torchvision", "adapts_input_shape": True,
             "adapts_num_classes": True},
            {"id": "imagenet1k_v2", "label": "V2", "task": "t", "dataset": "D",
             "input_shape": [3, 224, 224], "num_classes": 1000,
             "source": "https://x/y.pt", "adapts_input_shape": True,
             "adapts_num_classes": True},
        ],
    }
    config.update(extra)
    return config


class TestPreloadRegimeResolution:
    def test_explicit_weight_source_always_wins(self):
        plan = DeploymentPlan.resolve({"weight_source": "w.pt", "preload_weights": True})
        assert plan.weight_source == "w.pt"

    def test_preload_resolves_to_the_builders_default_weight_set(self):
        plan = DeploymentPlan.resolve(_sets_config(preload_weights=True))
        assert plan.weight_source == "torchvision"
        assert plan.pretrained_weight_set["id"] == "imagenet1k_v1"

    def test_a_chosen_weight_set_supplies_its_own_source(self):
        plan = DeploymentPlan.resolve(
            _sets_config(preload_weights=True, pretrained_weight_set="imagenet1k_v2")
        )
        assert plan.weight_source == "https://x/y.pt"
        assert plan.pretrained_weight_set["id"] == "imagenet1k_v2"

    def test_preload_without_a_registered_set_fails_loud(self):
        with pytest.raises(ValueError, match="pretrained_weight_sets"):
            DeploymentPlan.resolve({"preload_weights": True})

    def test_preload_on_a_builder_that_registers_nothing_fails_loud(self):
        with pytest.raises(ValueError, match="registers no pretrained weights"):
            DeploymentPlan.resolve(
                {"preload_weights": True, "pretrained_weight_sets": [],
                 "model_type": "lenet5"}
            )

    def test_no_regime_carries_no_weight_set(self):
        plan = DeploymentPlan.resolve(_sets_config())
        assert plan.weight_source is None
        assert plan.pretrained_weight_set is None


class TestIncompatibleWeightsNeverLoadSilently:
    """A set whose declared geometry / class count contradicts the data provider
    and which the builder cannot adapt is FILTERED from the legal set; a document
    that pins it fails loud naming the mismatch."""

    def _strict(self, **extra):
        config = {
            "pretrained_weight_sets": [
                {"id": "strict", "label": "S", "task": "t", "dataset": "D",
                 "input_shape": [3, 224, 224], "num_classes": 1000,
                 "source": "torchvision", "adapts_input_shape": False,
                 "adapts_num_classes": False},
            ],
            "input_shape": (1, 28, 28),
            "num_classes": 10,
            "model_type": "strict_builder",
        }
        config.update(extra)
        return config

    def test_the_run_refuses_a_geometry_incompatible_set(self):
        with pytest.raises(ValueError, match="no input adaptation"):
            DeploymentPlan.resolve(self._strict(preload_weights=True))

    def test_pinning_the_incompatible_set_names_the_mismatch(self):
        with pytest.raises(ValueError, match="cannot be used here"):
            DeploymentPlan.resolve(
                self._strict(preload_weights=True, pretrained_weight_set="strict")
            )

    def test_an_adapting_set_loads_on_a_different_workload(self):
        """The tier-1/2 corpus deploys ImageNet weights onto CIFAR-10 — the
        torchvision builders adapt, so this must keep resolving."""
        plan = DeploymentPlan.resolve(_sets_config(
            preload_weights=True, input_shape=(3, 32, 32), num_classes=10,
        ))
        assert plan.weight_source == "torchvision"


class TestWeightSourceResolution:
    """ONE resolution of the pretrained-weight concept, shared by the
    DeploymentPlan and the wizard — explicit declaration > the chosen registered
    weight set's own source > from-scratch."""

    def test_explicit_declaration_wins(self):
        assert resolve_weight_source(
            _sets_config(weight_source="/ckpt.pt", preload_weights=True),
        ) == "/ckpt.pt"

    def test_regime_resolves_the_builder_registration(self):
        assert resolve_weight_source(_sets_config(preload_weights=True)) == "torchvision"

    def test_no_regime_means_from_scratch(self):
        assert resolve_weight_source(_sets_config()) is None
        assert resolve_weight_source({}) is None

    def test_regime_without_a_registration_fails_loud(self):
        with pytest.raises(ValueError, match="registers no pretrained weight set"):
            resolve_weight_source({"preload_weights": True})
