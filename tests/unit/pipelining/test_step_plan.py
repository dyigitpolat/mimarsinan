"""Vector V5: contract-driven StepPlan + per-step ``applies_to(plan)``.

Locks the new seam: each ``PipelineStep`` owns its applicability, and a
``StepPlan`` filters an ordered registry. The dispatch/precedence rules that
used to be an 80-line conditional in ``get_pipeline_step_specs`` are now
explicit on the steps — these tests make them assertable.
"""

import pytest

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.step_plan import (
    StepPlan,
    StepSpec,
    StepPlanContractError,
)
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.core.pipelines.deployment_specs import (
    get_pipeline_step_specs,
    _STEP_PLAN,
)
from mimarsinan.pipelining.pipeline_steps import (
    ActivationAdaptationStep,
    ActivationAnalysisStep,
    ActivationQuantizationStep,
    ActivationShiftStep,
    ArchitectureSearchStep,
    ClampAdaptationStep,
    CoreQuantizationVerificationStep,
    LIFAdaptationStep,
    ModelBuildingStep,
    ModelConfigurationStep,
    NoiseAdaptationStep,
    PretrainingStep,
    PruningAdaptationStep,
    QuantizationVerificationStep,
    TTFSCycleAdaptationStep,
    TorchMappingStep,
    WeightPreloadingStep,
    WeightQuantizationStep,
)


def _plan(**overrides) -> DeploymentPlan:
    cfg = {"configuration_mode": "user", "spiking_mode": "lif", "model_type": "mlp_mixer"}
    cfg.update(overrides)
    return DeploymentPlan.resolve(cfg)


# ── StepPlan mechanics ───────────────────────────────────────────────────────

class TestStepPlanMechanics:
    def test_resolve_keeps_registry_order_and_drops_non_applicable(self):
        class _A(PipelineStep):
            @classmethod
            def applies_to(cls, plan):
                return True

        class _B(PipelineStep):
            @classmethod
            def applies_to(cls, plan):
                return False

        sp = StepPlan([StepSpec("A", _A), StepSpec("B", _B), StepSpec("A2", _A)])
        assert [s.to_pair() for s in sp.resolve(_plan())] == [("A", _A), ("A2", _A)]

    def test_callable_entry_is_spliced_in_order(self):
        class _A(PipelineStep):
            pass

        sp = StepPlan([
            StepSpec("A", _A),
            lambda plan: [StepSpec("X", _A), StepSpec("Y", _A)],
        ])
        assert [s.to_pair() for s in sp.resolve(_plan())] == [("A", _A), ("X", _A), ("Y", _A)]

    def test_step_spec_defaults_to_class_applies_to(self):
        spec = StepSpec("Pruning", PruningAdaptationStep)
        assert spec.applies_to(_plan(pruning=True, pruning_fraction=0.3)) is True
        assert spec.applies_to(_plan()) is False

    def test_explicit_applies_override_wins_over_class_method(self):
        spec = StepSpec("Pruning", PruningAdaptationStep, applies=lambda p: True)
        assert spec.applies_to(_plan()) is True  # class method would be False


# ── base default ─────────────────────────────────────────────────────────────

class TestBaseApplicability:
    def test_base_pipeline_step_always_applies(self):
        assert PipelineStep.applies_to(_plan()) is True

    def test_unconditional_steps_inherit_true(self):
        assert ModelBuildingStep.applies_to(_plan()) is True


# ── per-step predicates (the dispatch rules, made explicit) ──────────────────

class TestConfigurationDispatch:
    def test_search_vs_fixed_are_mutually_exclusive(self):
        search = _plan(model_config_mode="search")
        fixed = _plan(model_config_mode="fixed")
        assert search.search_mode != "fixed"
        assert fixed.search_mode == "fixed"
        assert ArchitectureSearchStep.applies_to(search)
        assert not ModelConfigurationStep.applies_to(search)
        assert ModelConfigurationStep.applies_to(fixed)
        assert not ArchitectureSearchStep.applies_to(fixed)

    def test_preload_vs_pretrain_are_mutually_exclusive(self):
        preload = _plan(weight_source="w.pt")
        scratch = _plan()
        assert WeightPreloadingStep.applies_to(preload)
        assert not PretrainingStep.applies_to(preload)
        assert PretrainingStep.applies_to(scratch)
        assert not WeightPreloadingStep.applies_to(scratch)

    def test_torch_mapping_only_for_torch_models(self):
        # torch_custom / mlp_mixer register as category "torch"; simple_mlp is native.
        assert TorchMappingStep.applies_to(_plan(model_type="torch_custom"))
        assert TorchMappingStep.applies_to(_plan(model_type="mlp_mixer"))
        assert not TorchMappingStep.applies_to(_plan(model_type="simple_mlp"))

    def test_pruning_requires_fraction(self):
        assert PruningAdaptationStep.applies_to(_plan(pruning=True, pruning_fraction=0.3))
        assert not PruningAdaptationStep.applies_to(_plan(pruning=True, pruning_fraction=0.0))
        assert not PruningAdaptationStep.applies_to(_plan())


class TestActivationFamilyDispatch:
    """Cycle-accurate tuning keeps its own step but no longer skips preconditioning."""

    @pytest.mark.parametrize("mode,lif_style", [
        ("lif", True),
        ("ttfs_cycle_based", True),
        ("ttfs", False),
        ("ttfs_quantized", False),
    ])
    def test_is_lif_style_matches_policy(self, mode, lif_style):
        plan = _plan(spiking_mode=mode)
        assert plan.is_lif_style is lif_style
        assert plan.mode_policy().single_step_activation_replacement is lif_style

    def test_cycle_based_modes_run_preconditioning_before_tuning_step(self):
        lif = _plan(spiking_mode="lif")
        assert ActivationAdaptationStep.applies_to(lif)
        assert ClampAdaptationStep.applies_to(lif)
        assert ActivationShiftStep.applies_to(lif)
        assert ActivationQuantizationStep.applies_to(lif)
        assert LIFAdaptationStep.applies_to(lif)
        assert not TTFSCycleAdaptationStep.applies_to(lif)

        cyc = _plan(spiking_mode="ttfs_cycle_based")
        assert ActivationAdaptationStep.applies_to(cyc)
        assert ClampAdaptationStep.applies_to(cyc)
        assert ActivationShiftStep.applies_to(cyc)
        assert ActivationQuantizationStep.applies_to(cyc)
        assert TTFSCycleAdaptationStep.applies_to(cyc)  # cascaded default
        assert not LIFAdaptationStep.applies_to(cyc)

    def test_synchronized_skips_cycle_finetuning_but_keeps_preconditioning(self):
        # The synchronized floor-collapse keeps the floor NF from Activation
        # Quantization (the ttfs_quantized recovery) and does NOT swap to the ceil
        # segment driver, so TTFS Cycle Fine-Tuning is cascaded-only. Preconditioning
        # (clamp/shift/AQ) STILL runs — it installs the floor NF.
        sync = _plan(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized",
        )
        assert sync.is_synchronized_ttfs and not sync.is_cascaded_ttfs
        assert not TTFSCycleAdaptationStep.applies_to(sync)
        assert ActivationQuantizationStep.applies_to(sync)
        assert ClampAdaptationStep.applies_to(sync)
        assert ActivationShiftStep.applies_to(sync)

        cascaded = _plan(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded",
        )
        assert cascaded.is_cascaded_ttfs
        assert TTFSCycleAdaptationStep.applies_to(cascaded)

    def test_non_lif_style_picks_analytical_chain(self):
        ttfs = _plan(spiking_mode="ttfs")
        assert ActivationAdaptationStep.applies_to(ttfs)
        assert not LIFAdaptationStep.applies_to(ttfs)
        assert not TTFSCycleAdaptationStep.applies_to(ttfs)

    def test_noise_only_for_lif_style_when_enabled(self):
        assert NoiseAdaptationStep.applies_to(_plan(spiking_mode="lif", enable_training_noise=True))
        assert not NoiseAdaptationStep.applies_to(_plan(spiking_mode="lif", enable_training_noise=False))
        # non-LIF-style never gets noise, even if the flag is on.
        assert not NoiseAdaptationStep.applies_to(_plan(spiking_mode="ttfs", enable_training_noise=True))

    def test_clamp_for_cycle_based_ttfs_firing_or_act_quant(self):
        # Cycle-accurate tuning modes are preconditioned before their tuning step.
        assert ClampAdaptationStep.applies_to(_plan(spiking_mode="lif", activation_quantization=False))
        assert ClampAdaptationStep.applies_to(_plan(spiking_mode="ttfs_cycle_based", activation_quantization=False))
        # TTFS firing forces clamp even without act_q.
        assert ClampAdaptationStep.applies_to(_plan(spiking_mode="ttfs", activation_quantization=False))
        # act_q forces clamp (analytical ttfs without ttfs-firing path covered via act_q).
        assert ClampAdaptationStep.applies_to(_plan(spiking_mode="ttfs_quantized", activation_quantization=True))

    def test_activation_quant_chain_for_cycle_based_or_act_q(self):
        lif_default = _plan(spiking_mode="lif", activation_quantization=False)
        assert ActivationShiftStep.applies_to(lif_default)
        assert ActivationQuantizationStep.applies_to(lif_default)

        cyc_default = _plan(spiking_mode="ttfs_cycle_based", activation_quantization=False)
        assert ActivationShiftStep.applies_to(cyc_default)
        assert ActivationQuantizationStep.applies_to(cyc_default)

        on = _plan(spiking_mode="ttfs", activation_quantization=True)
        assert ActivationShiftStep.applies_to(on)
        assert ActivationQuantizationStep.applies_to(on)

        off = _plan(spiking_mode="ttfs", activation_quantization=False)
        assert not ActivationShiftStep.applies_to(off)
        assert not ActivationQuantizationStep.applies_to(off)


class TestWeightQuantizationDispatch:
    @pytest.mark.parametrize("step", [
        WeightQuantizationStep,
        QuantizationVerificationStep,
        CoreQuantizationVerificationStep,
    ])
    def test_weight_quant_steps_gate_on_flag(self, step):
        assert step.applies_to(_plan(weight_quantization=True))
        assert not step.applies_to(_plan(weight_quantization=False))


# ── registry integrity ──────────────────────────────────────────────────────

class TestRegistryIntegrity:
    def test_resolve_matches_public_entry_point(self):
        for overrides in [
            {},
            {"spiking_mode": "ttfs", "activation_quantization": True, "weight_quantization": True},
            {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "synchronized"},
            {"model_type": "torch_custom", "weight_source": "w.pt"},
            {"pruning": True, "pruning_fraction": 0.3, "enable_loihi_simulation": True},
        ]:
            cfg = {"configuration_mode": "user", "spiking_mode": "lif",
                   "model_type": "mlp_mixer", **overrides}
            plan = DeploymentPlan.resolve(cfg)
            assert [s.to_pair() for s in _STEP_PLAN.resolve(plan)] == get_pipeline_step_specs(cfg)

    def test_registry_step_classes_have_a_step_class_each(self):
        for cls in _STEP_PLAN.step_classes():
            assert hasattr(cls, "applies_to")


# ── class-level data-contract declarations (V5 finish) ───────────────────────

class TestClassLevelDataContract:
    """Each step declares requires/promises/updates/clears at the CLASS level.

    The instance ``self.requires`` etc. set in ``__init__`` must read the class
    declaration verbatim (so the DAG is validatable without instantiation).
    """

    def test_declared_contract_matches_instance_attrs(self):
        # A real (uninstantiated) step's class contract == the instance contract
        # built in its __init__ (byte-identical lift).
        class _FakePipeline:
            config: dict = {}

        for step_cls in [
            ModelBuildingStep, PretrainingStep, WeightPreloadingStep,
            TorchMappingStep, ActivationAnalysisStep, LIFAdaptationStep,
            ClampAdaptationStep, WeightQuantizationStep,
            ModelConfigurationStep, ArchitectureSearchStep,
        ]:
            req, prom, upd, clr = step_cls.declared_contract()
            inst = step_cls(_FakePipeline())
            assert list(inst.requires) == list(req), step_cls.__name__
            assert list(inst.promises) == list(prom), step_cls.__name__
            assert list(inst.updates) == list(upd), step_cls.__name__
            assert list(inst.clears) == list(clr), step_cls.__name__

    def test_base_step_declares_empty_contract(self):
        assert PipelineStep.declared_contract() == ((), (), (), ())

    def test_ttfs_cycle_static_contract_is_lower_bound(self):
        # The instance-specific opt-in (activation_scales) is NOT in the static
        # class contract; the flag-off instance matches the class declaration.
        class _FakePipeline:
            config = {"ttfs_scale_aware_boundaries": False}

        req, _, _, _ = TTFSCycleAdaptationStep.declared_contract()
        assert "activation_scales" not in req
        inst = TTFSCycleAdaptationStep(_FakePipeline())
        assert list(inst.requires) == list(req)

        class _FakePipelineOn:
            config = {"ttfs_scale_aware_boundaries": True}

        inst_on = TTFSCycleAdaptationStep(_FakePipelineOn())
        assert "activation_scales" in inst_on.requires
        # The opt-in extra is still itself satisfiable (Activation Analysis,
        # which is unconditional, promises it earlier).
        assert ActivationAnalysisStep.applies_to(_plan(spiking_mode="ttfs_cycle_based"))


# ── assembly-time requires/promises DAG validation ──────────────────────────

class TestDataContractDagValidation:
    """``StepPlan.validate_data_contract`` asserts a producer for every consume."""

    def test_valid_plan_returns_same_specs_as_resolve(self):
        for overrides in [
            {},
            {"spiking_mode": "ttfs", "activation_quantization": True, "weight_quantization": True},
            {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "synchronized"},
            {"model_type": "torch_custom", "weight_source": "w.pt"},
            {"pruning": True, "pruning_fraction": 0.3, "enable_loihi_simulation": True},
        ]:
            plan = _plan(**overrides)
            assert _STEP_PLAN.validate_data_contract(plan) == _STEP_PLAN.resolve(plan)

    def test_public_entry_point_validates_the_dag(self):
        # get_pipeline_step_specs routes through validate_data_contract; a valid
        # config returns the resolved sequence unchanged.
        cfg = {"configuration_mode": "user", "spiking_mode": "lif", "model_type": "mlp_mixer"}
        assert get_pipeline_step_specs(cfg) == [
            s.to_pair() for s in _STEP_PLAN.resolve(DeploymentPlan.resolve(cfg))
        ]

    def test_missing_producer_raises_naming_the_entry(self):
        class _Producer(PipelineStep):
            PROMISES = ("model",)

        class _Consumer(PipelineStep):
            REQUIRES = ("model", "ghost_entry")

        sp = StepPlan([StepSpec("P", _Producer), StepSpec("C", _Consumer)])
        with pytest.raises(StepPlanContractError) as exc:
            sp.validate_data_contract(_plan())
        msg = str(exc.value)
        assert "ghost_entry" in msg
        assert "'C'" in msg  # the consuming step is named

    def test_consumer_before_producer_is_rejected(self):
        # A later step's promise does not satisfy an earlier step's require.
        class _Producer(PipelineStep):
            PROMISES = ("late_entry",)

        class _Consumer(PipelineStep):
            REQUIRES = ("late_entry",)

        sp = StepPlan([StepSpec("C", _Consumer), StepSpec("P", _Producer)])
        with pytest.raises(StepPlanContractError):
            sp.validate_data_contract(_plan())

    def test_update_keeps_entry_available_downstream(self):
        class _Producer(PipelineStep):
            PROMISES = ("model",)

        class _Updater(PipelineStep):
            REQUIRES = ("model",)
            UPDATES = ("model",)

        class _Consumer(PipelineStep):
            REQUIRES = ("model",)

        sp = StepPlan([
            StepSpec("P", _Producer),
            StepSpec("U", _Updater),
            StepSpec("C", _Consumer),
        ])
        # No raise: update re-publishes the entry for downstream consumers.
        assert len(sp.validate_data_contract(_plan())) == 3

    def test_clear_retracts_entry_for_downstream(self):
        class _Producer(PipelineStep):
            PROMISES = ("scratch",)
            CLEARS = ("scratch",)

        class _Consumer(PipelineStep):
            REQUIRES = ("scratch",)

        sp = StepPlan([StepSpec("P", _Producer), StepSpec("C", _Consumer)])
        with pytest.raises(StepPlanContractError):
            sp.validate_data_contract(_plan())

    @pytest.mark.parametrize("overrides", [
        {},
        {"spiking_mode": "ttfs", "activation_quantization": True, "weight_quantization": True},
        {"spiking_mode": "ttfs_cycle_based"},
        {"spiking_mode": "ttfs_cycle_based", "ttfs_scale_aware_boundaries": True},
        {"model_type": "torch_custom", "weight_source": "w.pt"},
        {"pruning": True, "pruning_fraction": 0.3},
        {"enable_sanafe_simulation": True},
        {"weight_quantization": True},
    ])
    def test_real_registry_dag_is_satisfiable_across_configs(self, overrides):
        # The production registry must validate for every config cell — the
        # assembly-time mirror of the runtime Pipeline.verify() DAG check.
        plan = _plan(**overrides)
        specs = _STEP_PLAN.validate_data_contract(plan)
        assert specs == _STEP_PLAN.resolve(plan)


class TestSemanticGroupsLiveOnTheRegistry:
    def test_every_registry_entry_declares_a_group(self):
        from mimarsinan.pipelining.core.pipelines.deployment_specs import _STEP_PLAN
        from mimarsinan.pipelining.core.step_plan import StepSpec

        for entry in _STEP_PLAN._entries:
            if isinstance(entry, StepSpec):
                assert entry.group != "other", entry.name

    def test_semantic_map_derives_from_registry(self):
        from conftest import default_config
        from mimarsinan.pipelining.core.pipelines.deployment_specs import (
            get_pipeline_semantic_group_by_step_name,
        )

        cfg = default_config()
        cfg["input_shape"] = (1, 8, 8)
        groups = get_pipeline_semantic_group_by_step_name(cfg)
        assert groups["Soft Core Mapping"] == "soft_mapping"
        assert groups["Pretraining"] == "pretraining"
        assert groups["Hard Core Mapping"] == "hardware"
        for name in groups:
            if "Simulation" in name:
                assert groups[name] == "simulation", name

    def test_parallel_group_dict_is_gone(self):
        from mimarsinan.pipelining.core.pipelines import deployment_specs

        assert not hasattr(deployment_specs, "_SEMANTIC_GROUP_BY_STEP_CLASS")
