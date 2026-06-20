"""Vector V5: contract-driven StepPlan + per-step ``applies_to(plan)``.

Locks the new seam: each ``PipelineStep`` owns its applicability, and a
``StepPlan`` filters an ordered registry. The dispatch/precedence rules that
used to be an 80-line conditional in ``get_pipeline_step_specs`` are now
explicit on the steps — these tests make them assertable.
"""

import pytest

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.step_plan import StepPlan, StepSpec
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.core.pipelines.deployment_specs import (
    get_pipeline_step_specs,
    _STEP_PLAN,
)
from mimarsinan.pipelining.pipeline_steps import (
    ActivationAdaptationStep,
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
        assert sp.resolve(_plan()) == [("A", _A), ("A2", _A)]

    def test_callable_entry_is_spliced_in_order(self):
        class _A(PipelineStep):
            pass

        sp = StepPlan([
            StepSpec("A", _A),
            lambda plan: [("X", _A), ("Y", _A)],
        ])
        assert sp.resolve(_plan()) == [("A", _A), ("X", _A), ("Y", _A)]

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
    """The LIF-style branch is composed from the V2 SpikingModePolicy."""

    @pytest.mark.parametrize("mode,lif_style", [
        ("lif", True),
        ("ttfs_cycle_based", True),
        ("rate", False),
        ("ttfs", False),
        ("ttfs_quantized", False),
    ])
    def test_is_lif_style_matches_policy(self, mode, lif_style):
        plan = _plan(spiking_mode=mode)
        assert plan.is_lif_style is lif_style
        assert plan.mode_policy().single_step_activation_replacement is lif_style

    def test_lif_style_picks_single_replacement_step(self):
        lif = _plan(spiking_mode="lif")
        assert LIFAdaptationStep.applies_to(lif)
        assert not TTFSCycleAdaptationStep.applies_to(lif)
        assert not ActivationAdaptationStep.applies_to(lif)

        cyc = _plan(spiking_mode="ttfs_cycle_based")
        assert TTFSCycleAdaptationStep.applies_to(cyc)
        assert not LIFAdaptationStep.applies_to(cyc)
        assert not ActivationAdaptationStep.applies_to(cyc)

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

    def test_clamp_for_ttfs_firing_or_act_quant_only_non_lif(self):
        # TTFS firing forces clamp even without act_q.
        assert ClampAdaptationStep.applies_to(_plan(spiking_mode="ttfs", activation_quantization=False))
        # act_q forces clamp.
        assert ClampAdaptationStep.applies_to(_plan(spiking_mode="rate", activation_quantization=True))
        # rate + no act_q → no clamp.
        assert not ClampAdaptationStep.applies_to(_plan(spiking_mode="rate", activation_quantization=False))
        # LIF-style never gets the chain.
        assert not ClampAdaptationStep.applies_to(_plan(spiking_mode="lif", activation_quantization=True))

    def test_activation_quant_chain_only_non_lif_with_act_q(self):
        on = _plan(spiking_mode="ttfs", activation_quantization=True)
        assert ActivationShiftStep.applies_to(on)
        assert ActivationQuantizationStep.applies_to(on)
        off = _plan(spiking_mode="ttfs", activation_quantization=False)
        assert not ActivationShiftStep.applies_to(off)
        assert not ActivationQuantizationStep.applies_to(off)
        lif = _plan(spiking_mode="lif", activation_quantization=True)
        assert not ActivationShiftStep.applies_to(lif)
        assert not ActivationQuantizationStep.applies_to(lif)


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
            assert _STEP_PLAN.resolve(plan) == get_pipeline_step_specs(cfg)

    def test_registry_step_classes_have_a_step_class_each(self):
        for cls in _STEP_PLAN.step_classes():
            assert hasattr(cls, "applies_to")
