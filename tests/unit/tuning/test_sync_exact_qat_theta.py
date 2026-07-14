"""sync exact-QAT theta-in-loop enhancement (default-off).

sync_exact_qat already trains the deployed ceil KERNEL, but with theta FROZEN
(no promote_theta_for_exact_qat, and TTFSCeilStaircaseDecorator applies theta
around a plain-STE staircase). This knob adds LIF-grade theta training: promote
theta trainable + swap to the gated TTFSCountStaircaseDecorator. Default off =
byte-identical (the frozen-theta ceil kernel); armed = theta trained in-loop.
"""

from __future__ import annotations

import torch.nn as nn

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
from mimarsinan.config_schema.registry import REGISTRY, FieldType
from mimarsinan.models.nn.decorators.clamp_quantize import (
    TTFSCeilStaircaseDecorator,
    TTFSCountStaircaseDecorator,
)


def _sync_cfg(*, theta=True, steps=8):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "synchronized"
    cfg["firing_mode"] = "TTFS"
    cfg["thresholding_mode"] = "<="
    cfg["simulation_steps"] = steps
    cfg["target_tq"] = steps
    cfg["sync_exact_qat"] = True
    if theta:
        cfg["sync_exact_qat_theta"] = True
    return cfg


class TestRegistry:
    def test_knob_is_registry_validated_bool_default_off(self):
        entry = REGISTRY["sync_exact_qat_theta"]
        assert entry.type is FieldType.BOOL
        assert entry.doc
        assert entry.derived_default({}) is False
        assert "sync_exact_qat_theta" in CONFIG_KEYS_SET


class TestDispatch:
    def _tuner_and_decorator(self, tmp_path, cfg):
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
        model = make_tiny_supermodel(hidden_layers=2)
        manager = create_adaptation_manager_for_model(cfg, model)
        tuner = ActivationQuantizationTuner(
            pipeline, model, cfg["target_tq"], 0.5, cfg["lr"], manager,
        )
        manager.quantization_rate = 1.0
        perceptron = list(model.get_perceptrons())[1]  # a non-encoder hop
        dec = manager.get_rate_adjusted_quantization_decorator(cfg, perceptron)
        return tuner, model, perceptron, dec

    def test_armed_uses_gated_decorator_and_trains_theta(self, tmp_path):
        tuner, model, perceptron, dec = self._tuner_and_decorator(
            tmp_path, _sync_cfg(theta=True))
        try:
            assert isinstance(dec.decorator, TTFSCountStaircaseDecorator)
            assert perceptron.activation_scale.requires_grad
        finally:
            tuner.close()

    def test_off_keeps_the_frozen_theta_ceil_kernel(self, tmp_path):
        tuner, model, perceptron, dec = self._tuner_and_decorator(
            tmp_path, _sync_cfg(theta=False))
        try:
            assert isinstance(dec.decorator, TTFSCeilStaircaseDecorator)
            assert not perceptron.activation_scale.requires_grad
        finally:
            tuner.close()

    def test_sync_theta_is_scalar_not_per_channel(self, tmp_path):
        # The synchronized mapper forward can't route per-channel theta, so sync
        # promotes a SCALAR-per-perceptron theta (per_channel=False).
        tuner, model, perceptron, dec = self._tuner_and_decorator(
            tmp_path, _sync_cfg(theta=True))
        try:
            for p in model.get_perceptrons():
                if getattr(p, "is_encoding_layer", False):
                    continue
                assert p.activation_scale.dim() == 0, p
        finally:
            tuner.close()


class TestScalarPromotion:
    def test_per_channel_false_promotes_all_scalar(self):
        from mimarsinan.spiking.theta_cotrain import promote_theta_for_exact_qat

        model = make_tiny_supermodel(hidden_layers=2)
        report = promote_theta_for_exact_qat(model, per_channel=False)
        assert report["per_channel"] == []
        for p in model.get_perceptrons():
            if getattr(p, "is_encoding_layer", False):
                continue
            assert p.activation_scale.requires_grad and p.activation_scale.dim() == 0
