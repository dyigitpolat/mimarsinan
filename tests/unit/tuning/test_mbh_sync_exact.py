"""Sync exact-kernel QAT endpoint — the default synchronized recipe (MBH X3, T6).

Driven by the ConversionPolicy recipe knob ``sync_exact_qat`` (folded into the
config for the synchronized schedule; knob off keeps the floor+half-step proxy
bit-identical — ttfs_quantized stays on the floor recipe, an X4 follow-up).
With the knob on AND the mode synchronized ttfs_cycle (deployment-contract
predicate), the AQ decorator stack trains the EXACT deployed composition: the
ceil TTFS kernel replaces the floor staircase + half-step ShiftDecorator proxy,
segment-entry perceptrons train through the deployed input grid snap, and the
mapping-time +0.5/Tq bias compensation — which exists solely to reconcile the
floor proxy — is skipped (explicitly, asserted) for models trained this way.
Every other mode is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.common import env
from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.models.spiking.wire_semantics import (
    ttfs_grid_quantize_np,
    ttfs_quantized_staircase_np,
)
from mimarsinan.tuning.orchestration.adaptation_manager import (
    AdaptationManager,
    install_sync_entry_grid_snap,
    model_trained_sync_exact,
    sync_exact_qat_active,
)
from mimarsinan.tuning.shift_calculation import calculate_activation_shift


def _sync_cfg(tq=4, steps=4, *, exact=True):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "synchronized"
    cfg["activation_quantization"] = True
    cfg["thresholding_mode"] = "<="
    cfg["target_tq"] = tq
    cfg["simulation_steps"] = steps
    if exact:
        cfg["sync_exact_qat"] = True
    return cfg


def _perceptron(theta=1.0):
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

    p = Perceptron(4, 8, normalization=nn.Identity())
    p.set_activation_scale(float(theta))
    return p


def _quant_decorator_output(cfg, perceptron, x):
    """Apply the manager's quantization decorator (rate 1.0) around identity."""
    manager = AdaptationManager()
    manager.quantization_rate = 1.0
    dec = manager.get_rate_adjusted_quantization_decorator(cfg, perceptron)
    act = TransformedActivation(nn.Identity(), [dec])
    with torch.no_grad():
        return act(x)


# A sweep hitting exact grid ties (k/S), the dead zone (x < theta/S), interiors,
# saturation (x >= theta), and negatives — all exactly representable in float32.
_SWEEP = [
    -0.5, 0.0, 0.01, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4375, 0.5,
    0.625, 0.75, 0.8, 0.875, 0.9, 1.0, 1.25, 2.0,
]


class TestRecipeKnob:
    def test_synchronized_recipe_turns_exact_endpoint_on(self):
        from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

        recipe = ConversionPolicy.derive("ttfs_cycle_based", "synchronized")
        assert recipe.knobs["sync_exact_qat"] is True
        assert recipe.special_case == "sync_exact_endpoint"

    def test_ttfs_quantized_recipe_stays_on_the_floor_proxy(self):
        # Green family, unchanged: promoting the exact endpoint for
        # ttfs_quantized is an X4 follow-up.
        from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

        assert "sync_exact_qat" not in ConversionPolicy.derive("ttfs_quantized").knobs


class TestKnobOffBitIdentity:
    def test_knob_off_predicate_is_false(self):
        assert sync_exact_qat_active(_sync_cfg(exact=False)) is False

    def test_knob_off_trains_floor_halfstep_proxy(self):
        cfg = _sync_cfg(tq=4, steps=4, exact=False)
        p = _perceptron(theta=1.0)
        x = torch.tensor([v for v in _SWEEP if v >= 0.0])
        y = _quant_decorator_output(cfg, p, x)
        delta = 1.0 / 4
        expected = torch.floor((x + delta / 2) / delta) * delta
        assert torch.equal(y, expected)
        assert not getattr(p, "_mbh_sync_exact_qat", False)

    def test_knob_off_no_grid_snap_install(self):
        model = make_tiny_supermodel()
        assert install_sync_entry_grid_snap(model, _sync_cfg(exact=False)) == 0
        for p in model.get_perceptrons():
            assert isinstance(p.input_activation, nn.Identity)


class TestKnobOnExactKernel:
    def test_predicate_requires_sync_schedule(self):
        assert sync_exact_qat_active(_sync_cfg()) is True
        for mode, schedule in [
            ("lif", None),
            ("ttfs", None),
            ("ttfs_quantized", None),
            ("ttfs_cycle_based", "cascaded"),
            ("ttfs_cycle_based", None),
        ]:
            cfg = _sync_cfg()
            cfg["spiking_mode"] = mode
            cfg["ttfs_cycle_schedule"] = schedule
            assert sync_exact_qat_active(cfg) is False, (mode, schedule)

    def test_trains_deployed_ceil_kernel_values(self):
        cfg = _sync_cfg(tq=4, steps=4)
        p = _perceptron(theta=1.0)
        x = torch.tensor(_SWEEP)
        y = _quant_decorator_output(cfg, p, x)
        expected = ttfs_quantized_staircase_np(
            np.asarray(_SWEEP, dtype=np.float64), 1.0, 4
        )
        np.testing.assert_array_equal(y.numpy().astype(np.float64), expected)
        assert getattr(p, "_mbh_sync_exact_qat", False)

    def test_kernel_uses_simulation_steps_not_tq(self):
        cfg = _sync_cfg(tq=4, steps=8)
        p = _perceptron(theta=1.0)
        x = torch.tensor([0.0625, 0.125, 0.1875, 0.5])
        y = _quant_decorator_output(cfg, p, x)
        expected = ttfs_quantized_staircase_np(
            x.numpy().astype(np.float64), 1.0, 8
        )
        np.testing.assert_array_equal(y.numpy().astype(np.float64), expected)

    def test_scaled_theta_matches_deployed_kernel(self):
        theta = 2.5
        cfg = _sync_cfg(tq=4, steps=4)
        p = _perceptron(theta=theta)
        x = torch.tensor([v * theta for v in _SWEEP])
        y = _quant_decorator_output(cfg, p, x)
        expected = theta * ttfs_quantized_staircase_np(
            x.numpy().astype(np.float64), theta, 4
        )
        np.testing.assert_allclose(y.numpy().astype(np.float64), expected, atol=1e-6)

    def test_ste_gradient_is_identity_on_interiors(self):
        cfg = _sync_cfg(tq=4, steps=4)
        p = _perceptron(theta=1.0)
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        dec = manager.get_rate_adjusted_quantization_decorator(cfg, p)
        act = TransformedActivation(nn.Identity(), [dec])
        x = torch.tensor([0.3, 0.6, 0.9], requires_grad=True)
        act(x).sum().backward()
        assert x.grad is not None
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_update_activation_installs_exact_kernel_and_marks(self):
        cfg = _sync_cfg(tq=4, steps=4)
        manager = AdaptationManager()
        manager.clamp_rate = 1.0
        manager.quantization_rate = 1.0
        p = _perceptron(theta=1.0)
        manager.update_activation(cfg, p)
        x = torch.tensor(_SWEEP)
        with torch.no_grad():
            y = p.activation(x)
        # LeakyGradReLU + clamp[0, theta] commute with the ceil kernel pointwise.
        expected = ttfs_quantized_staircase_np(
            np.asarray(_SWEEP, dtype=np.float64), 1.0, 4
        )
        np.testing.assert_array_equal(y.numpy().astype(np.float64), expected)
        assert getattr(p, "_mbh_sync_exact_qat", False)


class TestGridSnapInstall:
    def test_installs_on_segment_entries_only(self):
        from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer

        model = make_tiny_supermodel()
        n = install_sync_entry_grid_snap(model, _sync_cfg(steps=4))
        assert n == 1
        encoding, entry = list(model.get_perceptrons())
        assert encoding.is_encoding_layer
        assert isinstance(encoding.input_activation, nn.Identity)
        assert isinstance(entry.input_activation, TTFSInputGridQuantizer)
        assert entry.input_activation.T == 4

    def test_install_is_idempotent(self):
        from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer

        model = make_tiny_supermodel()
        cfg = _sync_cfg(steps=4)
        assert install_sync_entry_grid_snap(model, cfg) == 1
        assert install_sync_entry_grid_snap(model, cfg) == 0
        _, entry = list(model.get_perceptrons())
        assert isinstance(entry.input_activation, TTFSInputGridQuantizer)

    def test_snap_forward_matches_deployed_grid_quantize(self):
        model = make_tiny_supermodel()
        install_sync_entry_grid_snap(model, _sync_cfg(steps=4))
        _, entry = list(model.get_perceptrons())
        # Dead zone (x < 1/(2S)), interiors, half-step ties, top tie, saturation.
        x = torch.tensor([0.0, 0.05, 0.124, 0.126, 0.3, 0.5, 0.874, 0.876, 1.0])
        with torch.no_grad():
            y = entry.input_activation(x)
        expected = ttfs_grid_quantize_np(x.numpy().astype(np.float64), 4)
        np.testing.assert_allclose(y.numpy().astype(np.float64), expected, atol=1e-7)

    def test_aq_tuner_installs_snap_when_knob_on(self, tmp_path):
        from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        cfg = _sync_cfg()
        cfg["optimization_driver"] = "fast"
        cfg["manager_rate_fast_rates"] = [0.5, 1.0]
        cfg["manager_rate_fast_steps_per_rate"] = 1
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        manager = create_adaptation_manager_for_model(cfg, model)
        ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)
        _, entry = list(model.get_perceptrons())
        assert isinstance(entry.input_activation, TTFSInputGridQuantizer)

    def test_aq_tuner_knob_off_leaves_inputs_untouched(self, tmp_path):
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        cfg = _sync_cfg(exact=False)
        cfg["optimization_driver"] = "fast"
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        manager = create_adaptation_manager_for_model(cfg, model)
        ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)
        for p in model.get_perceptrons():
            assert isinstance(p.input_activation, nn.Identity)


class TestTrainedMarker:
    def test_unmarked_model_is_false(self):
        model = make_tiny_supermodel()
        assert model_trained_sync_exact(model) is False

    def test_fully_marked_model_is_true(self):
        cfg = _sync_cfg()
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            manager.update_activation(cfg, p)
        assert model_trained_sync_exact(model) is True

    def test_mixed_marking_fails_loud(self):
        cfg = _sync_cfg()
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        model = make_tiny_supermodel()
        first = next(iter(model.get_perceptrons()))
        manager.update_activation(cfg, first)
        with pytest.raises(AssertionError, match="sync-exact"):
            model_trained_sync_exact(model)


class TestBiasCompensationSkip:
    """The mapping-time +0.5/Tq bake is skipped iff the model trained the exact kernel."""

    def _run_step_helper(self, model, cfg, act_q=True):
        from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
            SoftCoreMappingStep,
        )

        pipeline = MockPipeline(config=cfg)
        fake_step = SimpleNamespace(pipeline=pipeline)
        SoftCoreMappingStep._apply_ttfs_quantization_bias_compensation(
            fake_step, model, act_q
        )

    def test_comp_applied_for_floor_trained_model(self):
        cfg = _sync_cfg(tq=4, exact=False)
        model = make_tiny_supermodel()
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        self._run_step_helper(model, cfg)
        for p, b in zip(model.get_perceptrons(), before):
            if getattr(p, "is_encoding_layer", False):
                assert torch.equal(p.layer.bias.detach(), b)
                continue
            assert getattr(p, "_ttfs_shift_baked_into_bias", False)
            shift = float(
                calculate_activation_shift(4, p.activation_scale)
                / p.activation_scale
            )
            np.testing.assert_allclose(
                p.layer.bias.detach().numpy(), (b + shift).numpy(), atol=1e-6,
            )

    def test_comp_skipped_for_exact_trained_model(self):
        cfg = _sync_cfg(tq=4)
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            manager.update_activation(cfg, p)
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        self._run_step_helper(model, cfg)
        for p, b in zip(model.get_perceptrons(), before):
            assert torch.equal(p.layer.bias.detach(), b)
            assert not getattr(p, "_ttfs_shift_baked_into_bias", False)

    def test_exact_marker_on_non_sync_plan_fails_loud(self):
        sync_cfg = _sync_cfg(tq=4)
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            manager.update_activation(sync_cfg, p)
        ttfsq_cfg = _sync_cfg(tq=4)
        ttfsq_cfg["spiking_mode"] = "ttfs_quantized"
        ttfsq_cfg["ttfs_cycle_schedule"] = None
        with pytest.raises(AssertionError):
            self._run_step_helper(model, ttfsq_cfg)


class TestComposition:
    def test_composes_with_mbh_ledger_flag(self, monkeypatch, tmp_path):
        from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        monkeypatch.setenv(env.MBH_LEDGER_VAR, "1")
        cfg = _sync_cfg()
        cfg["optimization_driver"] = "fast"
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        manager = create_adaptation_manager_for_model(cfg, model)
        tuner = ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)
        assert tuner is not None
        _, entry = list(model.get_perceptrons())
        assert isinstance(entry.input_activation, TTFSInputGridQuantizer)


class TestSyncEntryHalfStepCallSite:
    """[5v B1(ii)] the AQ tuner folds the entry half-step exactly once, only
    for sync exact-QAT runs with the recipe knob on."""

    def _tuner(self, tmp_path, cfg):
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_quantization_tuner import (
            ActivationQuantizationTuner,
        )

        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        manager = create_adaptation_manager_for_model(cfg, model)
        return ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)

    def test_knob_on_folds_non_encoding_hops(self, tmp_path, capsys):
        cfg = _sync_cfg()
        cfg["sync_entry_half_step"] = True
        tuner = self._tuner(tmp_path, cfg)
        try:
            assert "[MBH-B1] sync entry half-step folded" in capsys.readouterr().out
            flags = [
                bool(getattr(p, "_sync_entry_half_step_folded", False))
                for p in tuner.model.get_perceptrons()
            ]
            encoders = [
                bool(getattr(p, "is_encoding_layer", False))
                for p in tuner.model.get_perceptrons()
            ]
            for flag, is_enc in zip(flags, encoders):
                assert flag != is_enc
        finally:
            tuner.close()

    def test_knob_off_is_bit_identical(self, tmp_path):
        cfg = _sync_cfg()
        assert "sync_entry_half_step" not in cfg
        tuner = self._tuner(tmp_path, cfg)
        try:
            assert not any(
                getattr(p, "_sync_entry_half_step_folded", False)
                for p in tuner.model.get_perceptrons()
            )
        finally:
            tuner.close()

    def test_non_sync_mode_never_folds(self, tmp_path):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_quantized"
        cfg["activation_quantization"] = True
        cfg["optimization_driver"] = "fast"
        cfg["sync_entry_half_step"] = True  # accidental knob: mode gate wins
        tuner = self._tuner(tmp_path, cfg)
        try:
            assert not any(
                getattr(p, "_sync_entry_half_step_folded", False)
                for p in tuner.model.get_perceptrons()
            )
        finally:
            tuner.close()
