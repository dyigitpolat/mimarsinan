"""LIF exact-QAT KD-teacher lever (lif_exact_qat_program.md §8, WS-Z).

Config knob ``lif_exact_qat_kd`` (registry-validated, default OFF; config-armable
— recipe arming is a probe-A/B follow-up): the AQ-hosted exact-QAT trains with
plain CE on-pipeline (the measured WORST KD arm, −1.70 SEd), so the endpoint
saturates at the AQ envelope. This knob distils it to the POST-STRUCTURAL float
teacher (a Reference Teacher Snapshot step captures it after Scale Migration),
which the exact-QAT endpoint saturates at instead — WS-Z-measured WIN (>1 SE at
both S=4 cells). The knob pairs with ``lif_exact_qat``: it downgrades with the
exact arm (Novena/opt-out) and an explicit contradiction fails loud.
"""

from __future__ import annotations

import torch.nn as nn
import pytest

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters
from mimarsinan.config_schema.recipe_fold import _pair_lif_exact_qat_kd
from mimarsinan.config_schema.registry import REGISTRY, FieldType
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.tuning.orchestration.blend_ramp import KDClassificationLoss
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy
from mimarsinan.tuning.orchestration.lif_exact_qat import lif_exact_qat_kd_active


def _lif_cfg(*, kd=True, steps=8):
    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["activation_quantization"] = True
    cfg["weight_quantization"] = True
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = steps
    cfg["cycle_accurate_lif_forward"] = True
    if kd:
        cfg["lif_exact_qat_kd"] = True
    return cfg


class TestRegistryAndRecipe:
    def test_knob_is_registry_validated_bool_default_off(self):
        entry = REGISTRY["lif_exact_qat_kd"]
        assert entry.type is FieldType.BOOL
        assert entry.doc
        assert entry.derived_default is not None
        assert entry.derived_default({}) is False
        assert "lif_exact_qat_kd" in CONFIG_KEYS_SET

    def test_recipe_does_not_arm_it(self):
        # Arming is a probe-A/B follow-up; the knob is config-armable only.
        for mode, schedule in [
            ("lif", None), ("ttfs", None), ("ttfs_quantized", None),
            ("ttfs_cycle_based", "cascaded"), ("ttfs_cycle_based", "synchronized"),
        ]:
            assert "lif_exact_qat_kd" not in ConversionPolicy.derive(mode, schedule).knobs


class TestDerivation:
    def test_armed_with_exact_qat_recipe_default_stays_on(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "lif_exact_qat_kd": True}
        derive_deployment_parameters(dp)
        assert dp["lif_exact_qat"] is True
        assert dp["lif_exact_qat_kd"] is True

    def test_non_lif_mode_fails_loud(self):
        dp = {"spiking_mode": "ttfs_quantized", "weight_quantization": True,
              "lif_exact_qat_kd": True}
        with pytest.raises(ValueError, match="lif_exact_qat"):
            derive_deployment_parameters(dp)

    def test_explicit_kd_on_novena_fails_loud(self):
        # The recipe-default exact arm downgrades on Novena; an explicit KD arm
        # that now has no exact arm to ride is a contradiction.
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "firing_mode": "Novena", "lif_exact_qat_kd": True}
        with pytest.raises(ValueError, match="lif_exact_qat"):
            derive_deployment_parameters(dp)

    def test_explicit_exact_qat_off_fails_loud(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "lif_exact_qat": False, "lif_exact_qat_kd": True}
        with pytest.raises(ValueError, match="lif_exact_qat"):
            derive_deployment_parameters(dp)

    def test_absent_is_byte_identical(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True}
        derive_deployment_parameters(dp)
        assert "lif_exact_qat_kd" not in dp


class TestFoldDowngradeCoupling:
    def test_recipe_default_kd_downgrades_silently_when_exact_off(self):
        # Simulate a FUTURE recipe-injected KD (not document-explicit): when the
        # exact arm resolves off, the KD arm downgrades with it — no raise.
        dp = {"spiking_mode": "lif", "lif_exact_qat": False,
              "lif_exact_qat_kd": True}
        _pair_lif_exact_qat_kd(dp, explicit=set())
        assert dp["lif_exact_qat_kd"] is False

    def test_explicit_kd_without_exact_raises(self):
        dp = {"spiking_mode": "lif", "lif_exact_qat": False,
              "lif_exact_qat_kd": True}
        with pytest.raises(ValueError, match="lif_exact_qat"):
            _pair_lif_exact_qat_kd(dp, explicit={"lif_exact_qat_kd"})

    def test_off_is_noop(self):
        dp = {"spiking_mode": "lif", "lif_exact_qat": True}
        _pair_lif_exact_qat_kd(dp, explicit=set())
        assert "lif_exact_qat_kd" not in dp


class TestPredicate:
    def test_off_is_false(self):
        assert lif_exact_qat_kd_active(_lif_cfg(kd=False)) is False

    def test_armed_pair_is_true(self):
        cfg = _lif_cfg()
        cfg["lif_per_hop_retiming"] = True
        cfg["lif_exact_qat"] = True
        assert lif_exact_qat_kd_active(cfg) is True

    def test_composes_the_exact_failloud(self):
        # KD on but the exact preconditions broken (unpaired retiming) → the
        # exact predicate fails loud through the KD predicate.
        cfg = _lif_cfg()
        cfg["lif_exact_qat"] = True
        cfg["lif_per_hop_retiming"] = False
        with pytest.raises(ValueError, match="lif_per_hop_retiming"):
            lif_exact_qat_kd_active(cfg)


class TestPlanPredicate:
    def test_enabled_follows_the_derived_pair(self):
        cfg = {"spiking_mode": "lif", "weight_quantization": True,
               "lif_exact_qat_kd": True}
        derive_deployment_parameters(cfg)
        assert lif_exact_qat_kd_active(DeploymentPlan.resolve(cfg).config) is True

    def test_disabled_by_default(self):
        cfg = {"spiking_mode": "lif", "weight_quantization": True}
        derive_deployment_parameters(cfg)
        assert lif_exact_qat_kd_active(DeploymentPlan.resolve(cfg).config) is False


class TestSnapshotStep:
    def _step_names(self, cfg):
        # Production derives the config (merge_pipeline_config) before the step
        # plan resolves; mirror that so the recipe arms the exact-QAT pair.
        from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
            get_pipeline_step_specs,
        )
        cfg = dict(cfg)
        derive_deployment_parameters(cfg)
        return [name for name, _ in get_pipeline_step_specs(cfg)]

    def test_applies_to_follows_the_gate(self):
        from mimarsinan.pipelining.pipeline_steps import ReferenceTeacherSnapshotStep

        off = {"spiking_mode": "lif", "weight_quantization": True}
        derive_deployment_parameters(off)
        assert not ReferenceTeacherSnapshotStep.applies_to(DeploymentPlan.resolve(off))
        on = {"spiking_mode": "lif", "weight_quantization": True,
              "lif_exact_qat_kd": True}
        derive_deployment_parameters(on)
        assert ReferenceTeacherSnapshotStep.applies_to(DeploymentPlan.resolve(on))

    def test_absent_from_plan_by_default(self):
        cfg = {"configuration_mode": "user", "spiking_mode": "lif",
               "weight_quantization": True, "model_type": "mlp_mixer"}
        assert "Reference Teacher Snapshot" not in self._step_names(cfg)

    def test_present_and_precedes_clamp_when_armed(self):
        cfg = {"configuration_mode": "user", "spiking_mode": "lif",
               "weight_quantization": True, "model_type": "mlp_mixer",
               "lif_exact_qat_kd": True}
        names = self._step_names(cfg)
        assert "Reference Teacher Snapshot" in names
        snap = names.index("Reference Teacher Snapshot")
        # Post-structural, pre-conversion: after Torch Mapping (and Pruning /
        # Scale Migration where they apply), before the first function-changing
        # activation step (Clamp Adaptation) and Activation Analysis.
        assert names.index("Torch Mapping") < snap < names.index("Clamp Adaptation")
        assert snap < names.index("Activation Analysis")

    def test_promises_the_teacher(self):
        from mimarsinan.pipelining.pipeline_steps import ReferenceTeacherSnapshotStep

        assert "reference_teacher_model" in ReferenceTeacherSnapshotStep.PROMISES


class TestAQStepConditionalRequires:
    def _step(self, cfg):
        from mimarsinan.pipelining.pipeline_steps import ActivationQuantizationStep

        pipeline = MockPipeline(config=cfg)
        return ActivationQuantizationStep(pipeline)

    def test_requires_teacher_when_armed(self):
        cfg = {"spiking_mode": "lif", "weight_quantization": True,
               "lif_exact_qat_kd": True}
        derive_deployment_parameters(cfg)
        assert "reference_teacher_model" in self._step(cfg).requires

    def test_no_teacher_requirement_by_default(self):
        cfg = {"spiking_mode": "lif", "weight_quantization": True}
        derive_deployment_parameters(cfg)
        assert "reference_teacher_model" not in self._step(cfg).requires


class TestTunerLossSwap:
    def _tuner(self, tmp_path, cfg, kd_teacher=None, hidden_layers=2):
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        cfg = dict(cfg)
        cfg["optimization_driver"] = "fast"
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel(hidden_layers=hidden_layers)
        manager = create_adaptation_manager_for_model(cfg, model)
        tuner = ActivationQuantizationTuner(
            pipeline, model, cfg["target_tq"], 0.5, cfg["lr"], manager,
            kd_teacher=kd_teacher,
        )
        return tuner, model, pipeline

    def _armed_cfg(self):
        cfg = _lif_cfg()
        cfg["lif_exact_qat"] = True
        cfg["lif_per_hop_retiming"] = True
        cfg["kd_ce_alpha"] = 0.5
        cfg["kd_temperature"] = 4.0
        return cfg

    def test_no_teacher_keeps_plain_ce(self, tmp_path):
        cfg = self._armed_cfg()
        tuner, _model, pipeline = self._tuner(tmp_path, cfg, kd_teacher=None)
        try:
            assert tuner.trainer.loss_function is pipeline.loss
        finally:
            tuner.close()

    def test_teacher_swaps_in_kd_loss_with_config_weights(self, tmp_path):
        cfg = self._armed_cfg()
        teacher = nn.Linear(4, 4)
        tuner, _model, _pipeline = self._tuner(tmp_path, cfg, kd_teacher=teacher)
        try:
            loss = tuner.trainer.loss_function
            assert isinstance(loss, KDClassificationLoss)
            assert loss.alpha == 0.5
            assert loss.temperature == 4.0
            # Teacher frozen: eval mode, no grad.
            assert not any(p.requires_grad for p in teacher.parameters())
        finally:
            tuner.close()

    def test_teacher_ignored_when_exact_arm_off(self, tmp_path):
        # KD teacher without the exact arm must not swap the loss (the lever is
        # only meaningful at the exact endpoint).
        cfg = _lif_cfg(kd=False)
        teacher = nn.Linear(4, 4)
        tuner, _model, pipeline = self._tuner(tmp_path, cfg, kd_teacher=teacher)
        try:
            assert tuner.trainer.loss_function is pipeline.loss
        finally:
            tuner.close()
