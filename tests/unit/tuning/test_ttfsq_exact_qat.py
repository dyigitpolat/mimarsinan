"""ttfsq exact-QAT arm — the TTFS analog of lif_exact_qat.

Config knob ``ttfsq_exact_qat`` (registry-validated, default OFF; recipe arming
is a probe-A/B follow-up): the AQ stage trains the exact deployed TTFS ceil
staircase with theta in-loop (``TTFSCountStaircaseDecorator``) instead of the
float shift + floor-quantize proxy. No per-hop re-timing (TTFS is analytical).
Knob off is byte-identical everywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
from mimarsinan.config_schema.registry import REGISTRY, FieldType
from mimarsinan.models.nn.decorators.clamp_quantize import (
    TTFSCountStaircaseDecorator,
    QuantizeDecorator,
)
from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy
from mimarsinan.tuning.orchestration.ttfs_exact_qat import (
    model_trained_ttfsq_exact,
    ttfsq_exact_qat_active,
)


def _ttfsq_cfg(*, exact=True, steps=8):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_quantized"
    cfg["firing_mode"] = "TTFS"
    cfg["thresholding_mode"] = "<="
    cfg["simulation_steps"] = steps
    cfg["target_tq"] = steps
    if exact:
        cfg["ttfsq_exact_qat"] = True
    return cfg


class TestRegistryAndRecipe:
    def test_knob_is_registry_validated_bool_default_off(self):
        entry = REGISTRY["ttfsq_exact_qat"]
        assert entry.type is FieldType.BOOL
        assert entry.doc
        assert entry.derived_default is not None
        assert entry.derived_default({}) is False
        assert "ttfsq_exact_qat" in CONFIG_KEYS_SET

    def test_recipe_does_not_arm_it(self):
        for mode, schedule in [
            ("lif", None), ("ttfs", None), ("ttfs_quantized", None),
            ("ttfs_cycle_based", "synchronized"),
        ]:
            assert "ttfsq_exact_qat" not in ConversionPolicy.derive(mode, schedule).knobs


class TestPredicate:
    def test_off_is_false(self):
        assert ttfsq_exact_qat_active(_ttfsq_cfg(exact=False)) is False

    def test_armed_is_true(self):
        assert ttfsq_exact_qat_active(_ttfsq_cfg()) is True

    def test_non_ttfsq_mode_fails_loud(self):
        cfg = _ttfsq_cfg()
        cfg["spiking_mode"] = "lif"
        with pytest.raises(ValueError, match="ttfs_quantized"):
            ttfsq_exact_qat_active(cfg)


class TestAQTunerArm:
    def _tuner(self, tmp_path, cfg, hidden_layers=2):
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
        )
        return tuner, model, manager

    def _install_decorator(self, manager, cfg, perceptron):
        manager.quantization_rate = 1.0
        return manager.get_rate_adjusted_quantization_decorator(cfg, perceptron)

    def test_arm_installs_ttfs_count_staircase_and_trainable_theta(self, tmp_path, capsys):
        cfg = _ttfsq_cfg(steps=8)
        tuner, model, manager = self._tuner(tmp_path, cfg)
        try:
            for perceptron in model.get_perceptrons():
                dec = self._install_decorator(manager, cfg, perceptron)
                # RateAdjustedDecorator wraps the exact ceil-staircase decorator.
                assert isinstance(dec.decorator, TTFSCountStaircaseDecorator)
                # theta trainable in-loop on non-encoders; encoder frozen (like LIF).
                is_encoder = bool(getattr(perceptron, "is_encoding_layer", False))
                assert perceptron.activation_scale.requires_grad is (not is_encoder)
            assert model_trained_ttfsq_exact(model)
        finally:
            tuner.close()

    def test_knob_off_uses_the_shift_quantize_proxy(self, tmp_path):
        cfg = _ttfsq_cfg(exact=False)
        tuner, model, manager = self._tuner(tmp_path, cfg)
        try:
            perceptron = list(model.get_perceptrons())[1]
            dec = self._install_decorator(manager, cfg, perceptron)
            # off: the plain shift+quantize proxy (no TTFSCountStaircaseDecorator),
            # theta frozen.
            assert not isinstance(
                getattr(dec, "decorator", None), TTFSCountStaircaseDecorator
            )
            assert not perceptron.activation_scale.requires_grad
        finally:
            tuner.close()


class TestWQThetaFreeze:
    """The WQ lattice must freeze the exact-QAT in-loop theta for ttfsq/sync too
    (left trainable, it floods the projection with degenerate-channel routing
    and OOMs the graph — the real S8 A/B failure, 2026-07-14)."""

    def _freeze(self, model, cfg):
        from types import SimpleNamespace

        from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
            WeightQuantizationStep,
        )

        fake_step = SimpleNamespace(pipeline=MockPipeline(config=cfg))
        WeightQuantizationStep._freeze_exact_qat_theta(fake_step, model)

    def _promoted_model(self):
        from mimarsinan.spiking.theta_cotrain import promote_theta_for_exact_qat

        model = make_tiny_supermodel(hidden_layers=2)
        promote_theta_for_exact_qat(model)
        return model

    def test_ttfsq_exact_freezes_trainable_theta(self):
        from mimarsinan.tuning.orchestration.ttfs_exact_qat import mark_ttfsq_exact_qat

        model = self._promoted_model()
        for p in model.get_perceptrons():
            mark_ttfsq_exact_qat(p)
        assert any(p.activation_scale.requires_grad for p in model.get_perceptrons())
        self._freeze(model, _ttfsq_cfg())
        assert all(not p.activation_scale.requires_grad for p in model.get_perceptrons())

    def test_sync_theta_freezes_trainable_theta(self):
        model = self._promoted_model()
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "synchronized"
        cfg["sync_exact_qat"] = True
        cfg["sync_exact_qat_theta"] = True
        assert any(p.activation_scale.requires_grad for p in model.get_perceptrons())
        self._freeze(model, cfg)
        assert all(not p.activation_scale.requires_grad for p in model.get_perceptrons())

    def test_plain_ttfsq_leaves_theta_untouched(self):
        model = self._promoted_model()  # promoted but NOT exact-marked
        self._freeze(model, _ttfsq_cfg(exact=False))
        assert any(p.activation_scale.requires_grad for p in model.get_perceptrons())
