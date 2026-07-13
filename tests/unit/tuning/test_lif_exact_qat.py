"""LIF exact-QAT arm (lif_exact_qat_program.md §6) — the (A+R5) composition.

Config knob ``lif_exact_qat`` (registry-validated, default OFF; recipe arming is
a follow-up decision): the AQ stage hosts the staircase exact-QAT — the exact
deployed count staircase ``theta*clamp(F(T*z/theta),0,T)/T`` under staircase-STE
with theta trainable in the loop — replacing the Shift bake + T-anneal ladder +
one-shot half-step fold displacement (measured −2.06 pp). Deployment is the
per-hop RE-TIMED pair: ``lif_per_hop_retiming`` is auto-paired (the trained
staircase IS the per-hop twin; staircase deploy WITHOUT re-timing is the
measured Goodhart hole, −2.5 pp). Knob off is byte-identical everywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters
from mimarsinan.config_schema.registry import REGISTRY, FieldType
from mimarsinan.mapping.support.bias_compensation import (
    LIF_HALF_STEP_FLAG,
    apply_lif_half_step_bias_compensation,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer
from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.models.spiking.wire_semantics import (
    lif_count_staircase,
    lif_count_staircase_np,
)
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy
from mimarsinan.tuning.orchestration.lif_exact_qat import (
    install_lif_entry_input_quantizers,
    install_lif_input_quantizer,
    lif_exact_qat_active,
    lif_subsumed_ladder_steps,
    model_trained_lif_exact,
)


def _lif_cfg(*, exact=True, steps=8, thresholding="<"):
    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["activation_quantization"] = True
    cfg["thresholding_mode"] = thresholding
    cfg["simulation_steps"] = steps
    cfg["cycle_accurate_lif_forward"] = True
    if exact:
        cfg["lif_exact_qat"] = True
        cfg["lif_per_hop_retiming"] = True
    return cfg


def _perceptron(theta=1.0):
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

    p = Perceptron(4, 8, normalization=nn.Identity())
    p.set_activation_scale(float(theta))
    return p


def _quant_decorator_output(cfg, perceptron, x):
    manager = AdaptationManager()
    manager.quantization_rate = 1.0
    dec = manager.get_rate_adjusted_quantization_decorator(cfg, perceptron)
    act = TransformedActivation(nn.Identity(), [dec])
    with torch.no_grad():
        return act(x)


# Exact binary fractions: grid ties (k/T), the dead zone, interiors, saturation,
# negatives — all f32/f64-exact so strict-vs-inclusive tie behavior is honest.
_SWEEP = [
    -0.5, 0.0, 0.03125, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5,
    0.625, 0.75, 0.875, 0.90625, 1.0, 1.25, 2.0,
]


class TestRegistryAndRecipe:
    def test_knob_is_registry_validated_bool_default_off(self):
        entry = REGISTRY["lif_exact_qat"]
        assert entry.type is FieldType.BOOL
        assert entry.doc
        assert entry.derived_default is not None
        assert entry.derived_default({}) is False
        assert "lif_exact_qat" in CONFIG_KEYS_SET

    def test_recipe_does_not_arm_it(self):
        # Arming is a follow-up decision (tier-0 A/B); the knob is config-armable.
        for mode, schedule in [
            ("lif", None), ("ttfs", None), ("ttfs_quantized", None),
            ("ttfs_cycle_based", "cascaded"), ("ttfs_cycle_based", "synchronized"),
        ]:
            assert "lif_exact_qat" not in ConversionPolicy.derive(mode, schedule).knobs


class TestPairingDerivation:
    def test_armed_knob_auto_pairs_retiming(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "lif_exact_qat": True}
        derive_deployment_parameters(dp)
        assert dp["lif_per_hop_retiming"] is True

    def test_explicit_retiming_off_fails_loud(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "lif_exact_qat": True, "lif_per_hop_retiming": False}
        with pytest.raises(ValueError, match="lif_per_hop_retiming"):
            derive_deployment_parameters(dp)

    def test_non_lif_mode_fails_loud(self):
        dp = {"spiking_mode": "ttfs_quantized", "weight_quantization": True,
              "lif_exact_qat": True}
        with pytest.raises(ValueError, match="lif_exact_qat"):
            derive_deployment_parameters(dp)

    def test_novena_fails_loud(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "firing_mode": "Novena", "lif_exact_qat": True}
        with pytest.raises(ValueError, match="Novena|Default"):
            derive_deployment_parameters(dp)

    def test_knob_off_derivation_is_byte_identical(self):
        base = {"spiking_mode": "lif", "weight_quantization": True}
        dp = dict(base)
        derive_deployment_parameters(dp)
        assert "lif_exact_qat" not in dp
        assert dp["lif_per_hop_retiming"] is False


class TestPredicate:
    def test_off_is_false(self):
        assert lif_exact_qat_active(_lif_cfg(exact=False)) is False

    def test_on_and_paired_is_true(self):
        assert lif_exact_qat_active(_lif_cfg()) is True

    def test_non_lif_mode_is_false(self):
        cfg = _lif_cfg()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        assert lif_exact_qat_active(cfg) is False

    def test_unpaired_retiming_raises(self):
        cfg = _lif_cfg()
        cfg["lif_per_hop_retiming"] = False
        with pytest.raises(ValueError, match="lif_per_hop_retiming"):
            lif_exact_qat_active(cfg)

    def test_novena_raises(self):
        cfg = _lif_cfg()
        cfg["firing_mode"] = "Novena"
        with pytest.raises(ValueError, match="Default"):
            lif_exact_qat_active(cfg)

    def test_value_domain_forward_raises(self):
        cfg = _lif_cfg()
        cfg["cycle_accurate_lif_forward"] = False
        with pytest.raises(ValueError, match="cycle"):
            lif_exact_qat_active(cfg)


class TestKernelPair:
    """The commutation kernel: torch/np twins, both comparators (§6.4(i))."""

    @pytest.mark.parametrize("compare", ["<", "<="])
    @pytest.mark.parametrize("steps", [4, 8, 16, 32])
    def test_torch_matches_numpy(self, compare, steps):
        z = torch.tensor(_SWEEP, dtype=torch.float64)
        got = lif_count_staircase(
            z, torch.tensor(1.0, dtype=torch.float64), steps, compare_mode=compare,
        )
        expected = lif_count_staircase_np(
            np.asarray(_SWEEP, dtype=np.float64), 1.0, steps, compare_mode=compare,
        )
        np.testing.assert_array_equal(got.numpy(), expected)

    def test_inclusive_is_floor_and_strict_drops_exact_ties(self):
        # F = floor for '<='; F = ceil - 1 for strict '<' (Theorem 2 / A2).
        z = torch.tensor([0.25, 0.5], dtype=torch.float64)  # exact k/T ties, T=4
        inclusive = lif_count_staircase(z, torch.tensor(1.0).double(), 4, compare_mode="<=")
        strict = lif_count_staircase(z, torch.tensor(1.0).double(), 4, compare_mode="<")
        np.testing.assert_array_equal(inclusive.numpy(), [0.25, 0.5])
        np.testing.assert_array_equal(strict.numpy(), [0.0, 0.25])

    @pytest.mark.parametrize("compare", ["<", "<="])
    @pytest.mark.parametrize("steps", [4, 8, 16, 32])
    def test_a2_lock_kernel_equals_lif_constant_drive(self, compare, steps):
        # Exact-binary drives; strict '<' probes OFF exact grid ties: the
        # vendored IFNode eval path fires ``v >= threshold`` at a bit-exact
        # tie (the V9 lattice hazard, measure-zero on real data, P-L3) while
        # the deployed strict comparator — and the kernel — drops one level
        # (pinned in test_inclusive_is_floor_and_strict_drops_exact_ties).
        theta = 1.0
        off_tie = [(2 * k + 1) / (2 * steps) for k in range(-1, steps + 1)] + [
            (4 * k + 1) / (4 * steps) for k in range(steps)
        ]
        ties = [k / steps for k in range(-2, steps + 3)]
        values = off_tie if compare == "<" else off_tie + ties
        z = torch.tensor(values, dtype=torch.float32)
        lif = LIFActivation(T=steps, activation_scale=theta, thresholding_mode=compare)
        lif.eval()
        with torch.no_grad():
            node_out = lif(z)
        kernel_out = lif_count_staircase(
            z, torch.tensor(theta), steps, compare_mode=compare,
        )
        assert torch.equal(node_out, kernel_out.to(node_out.dtype))

    def test_a2_lock_per_channel_theta(self):
        steps = 8
        theta = torch.tensor([0.5, 1.0, 2.0, 4.0])
        z = torch.stack([theta * (k / steps) for k in range(steps + 2)])
        lif = LIFActivation(T=steps, activation_scale=theta, thresholding_mode="<=")
        lif.eval()
        with torch.no_grad():
            node_out = lif(z)
        kernel_out = lif_count_staircase(z, theta, steps, compare_mode="<=")
        assert torch.equal(node_out, kernel_out.to(node_out.dtype))


class TestSTEGradients:
    """Staircase-STE z-grad + the LSQ theta gradient q(r) − r·1[in-band] (§4.2)."""

    def _forward(self, z, theta, steps=8, compare="<"):
        from mimarsinan.models.nn.activations.autograd import LIFCountStaircaseFunction

        return LIFCountStaircaseFunction.apply(z, theta, steps, compare == "<")

    def test_z_grad_is_clamp_gated_identity(self):
        z = torch.tensor([-0.5, 0.09, 0.5, 0.99, 1.5], requires_grad=True)
        theta = torch.tensor(1.0)
        self._forward(z, theta).sum().backward()
        assert z.grad is not None
        # In-band (0 < r < 1) passes identity — including the count-0 dead zone
        # r < 1/(2T) (no dead-neuron trap); out-of-band is silent.
        assert torch.equal(z.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))

    def test_theta_grad_matches_lsq_formula(self):
        steps = 8
        z = torch.tensor([-0.5, 0.3, 0.7, 1.5])
        theta = torch.tensor(1.0, requires_grad=True)
        self._forward(z, theta, steps=steps).sum().backward()
        r = z / 1.0
        q = (torch.ceil(steps * r) - 1.0).clamp(0.0, float(steps)) / steps
        inband = ((r > 0) & (r < 1)).float()
        expected = (q - r * inband).sum()
        assert theta.grad is not None
        assert torch.allclose(theta.grad, expected)
        # Saturated channels (r >= 1) push theta UP with the full signal: q = 1.
        assert float(q[3]) == 1.0

    def test_per_channel_theta_grad_reduces_to_channel_shape(self):
        theta = torch.tensor([1.0, 2.0], requires_grad=True)
        z = torch.rand(5, 2)
        self._forward(z, theta).sum().backward()
        assert theta.grad is not None
        assert theta.grad.shape == theta.shape


class TestManagerInstall:
    def test_knob_on_installs_exact_count_staircase_and_marks(self):
        cfg = _lif_cfg(steps=8, thresholding="<")
        p = _perceptron(theta=1.0)
        x = torch.tensor(_SWEEP)
        y = _quant_decorator_output(cfg, p, x)
        expected = lif_count_staircase_np(
            np.asarray(_SWEEP, dtype=np.float64), 1.0, 8, compare_mode="<",
        )
        np.testing.assert_allclose(y.double().numpy(), expected, atol=1e-7)
        assert getattr(p, "_mbh_lif_exact_qat", False)

    def test_inclusive_comparator_routes_the_floor_form(self):
        cfg = _lif_cfg(steps=4, thresholding="<=")
        p = _perceptron(theta=1.0)
        x = torch.tensor(_SWEEP)
        y = _quant_decorator_output(cfg, p, x)
        expected = lif_count_staircase_np(
            np.asarray(_SWEEP, dtype=np.float64), 1.0, 4, compare_mode="<=",
        )
        np.testing.assert_allclose(y.double().numpy(), expected, atol=1e-7)

    def test_scaled_theta_matches_kernel(self):
        theta = 2.5
        cfg = _lif_cfg(steps=8)
        p = _perceptron(theta=theta)
        x = torch.tensor([v * theta for v in _SWEEP])
        y = _quant_decorator_output(cfg, p, x)
        expected = lif_count_staircase_np(
            x.double().numpy(), theta, 8, compare_mode="<",
        )
        np.testing.assert_allclose(y.double().numpy(), expected, atol=1e-6)

    def test_knob_off_keeps_the_lif_subsumption_byte_identical(self):
        cfg = _lif_cfg(exact=False)
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        p = _perceptron()
        manager.update_activation(cfg, p)
        act = p.activation
        assert isinstance(act, TransformedActivation)
        assert act.decorators == []
        assert not getattr(p, "_mbh_lif_exact_qat", False)

    def test_update_activation_installs_under_the_arm(self):
        cfg = _lif_cfg(steps=8)
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        p = _perceptron(theta=1.0)
        manager.update_activation(cfg, p)
        x = torch.tensor(_SWEEP)
        with torch.no_grad():
            y = p.activation(x)
        # LeakyGradReLU commutes with the count staircase pointwise on z >= 0
        # and both are 0 on z < 0.
        expected = lif_count_staircase_np(
            np.asarray(_SWEEP, dtype=np.float64), 1.0, 8, compare_mode="<",
        )
        np.testing.assert_allclose(y.double().numpy(), expected, atol=1e-7)

    def test_lif_active_finalize_subsumes_even_under_the_arm(self):
        cfg = _lif_cfg()
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        manager.lif_active = True
        p = _perceptron()
        manager.update_activation(cfg, p)
        act = p.activation
        assert isinstance(act, TransformedActivation)
        assert act.decorators == []

    def test_ladder_steps_unsubsumed_under_the_arm(self):
        on, off = _lif_cfg(), _lif_cfg(exact=False)
        assert lif_subsumed_ladder_steps(on, "quantization_rate", 96) == 96
        assert lif_subsumed_ladder_steps(on, "clamp_rate", 96) == 0
        assert lif_subsumed_ladder_steps(off, "quantization_rate", 96) == 0
        assert lif_subsumed_ladder_steps(off, "clamp_rate", 96) == 0


class TestTrainedMarker:
    def test_unmarked_model_is_false(self):
        assert model_trained_lif_exact(make_tiny_supermodel()) is False

    def test_fully_marked_model_is_true(self):
        cfg = _lif_cfg()
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            manager.update_activation(cfg, p)
        assert model_trained_lif_exact(model) is True

    def test_mixed_marking_fails_loud(self):
        cfg = _lif_cfg()
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        model = make_tiny_supermodel()
        manager.update_activation(cfg, next(iter(model.get_perceptrons())))
        with pytest.raises(AssertionError, match="lif-exact"):
            model_trained_lif_exact(model)


class TestEntryInputQuantizers:
    def test_installs_on_encoders_only_idempotently(self):
        model = make_tiny_supermodel()
        cfg = _lif_cfg(steps=8)
        assert install_lif_entry_input_quantizers(model, cfg) == 1
        assert install_lif_entry_input_quantizers(model, cfg) == 0
        perceptrons = list(model.get_perceptrons())
        assert perceptrons[0].is_encoding_layer
        assert isinstance(perceptrons[0].input_activation, ChipInputQuantizer)
        assert perceptrons[0].input_activation.T == 8
        for p in perceptrons[1:]:
            assert isinstance(p.input_activation, nn.Identity)

    def test_knob_off_installs_nothing(self):
        model = make_tiny_supermodel()
        assert install_lif_entry_input_quantizers(model, _lif_cfg(exact=False)) == 0
        for p in model.get_perceptrons():
            assert isinstance(p.input_activation, nn.Identity)

    def test_shared_installer_is_idempotent_per_perceptron(self):
        p = _perceptron()
        p.is_encoding_layer = True
        assert install_lif_input_quantizer(p, 8) is True
        assert install_lif_input_quantizer(p, 8) is False
        assert isinstance(p.input_activation, ChipInputQuantizer)


class TestShiftBakeSkip:
    def _tuner(self, tmp_path, cfg):
        from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
            create_adaptation_manager_for_model,
        )
        from mimarsinan.tuning.tuners.activation_shift_tuner import ActivationShiftTuner

        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        manager = create_adaptation_manager_for_model(cfg, model)
        tuner = ActivationShiftTuner(pipeline, model, 0.5, cfg["lr"], manager)
        return tuner, model, manager

    def test_arm_skips_the_unflagged_bias_bake(self, tmp_path):
        cfg = _lif_cfg()
        tuner, model, manager = self._tuner(tmp_path, cfg)
        try:
            before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
            tuner._apply_shift()
            for p, b in zip(model.get_perceptrons(), before):
                assert torch.equal(p.layer.bias.detach(), b)
            assert manager.shift_rate == 0.0
        finally:
            tuner.close()

    def test_knob_off_bakes_as_shipped(self, tmp_path):
        cfg = _lif_cfg(exact=False)
        tuner, model, manager = self._tuner(tmp_path, cfg)
        try:
            before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
            tuner._apply_shift()
            changed = any(
                not torch.equal(p.layer.bias.detach(), b)
                for p, b in zip(model.get_perceptrons(), before)
            )
            assert changed
            assert manager.shift_rate == 1.0
        finally:
            tuner.close()


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
        return tuner, model

    def test_arm_installs_fold_snaps_and_trainable_theta(self, tmp_path, capsys):
        cfg = _lif_cfg(steps=8)
        tuner, model = self._tuner(tmp_path, cfg)
        try:
            out = capsys.readouterr().out
            assert "[LIF-EXACT-QAT]" in out
            perceptrons = list(model.get_perceptrons())
            encoder, hidden, head = perceptrons
            # P-L6: the half-step folds ONCE at the install, encoders excluded.
            assert not getattr(encoder, LIF_HALF_STEP_FLAG, False)
            assert getattr(hidden, LIF_HALF_STEP_FLAG, False)
            assert getattr(head, LIF_HALF_STEP_FLAG, False)
            # Entry snap on the encoder (the deployed entry round, §4.3).
            assert isinstance(encoder.input_activation, ChipInputQuantizer)
            # Theta in-loop: encoder frozen; matching-axis hop per-channel;
            # externally-consumed head scalar-trainable (§6.2 seam constraints).
            assert not encoder.activation_scale.requires_grad
            assert hidden.activation_scale.requires_grad
            assert hidden.activation_scale.numel() == hidden.layer.weight.shape[0]
            assert head.activation_scale.requires_grad
            assert head.activation_scale.dim() == 0
        finally:
            tuner.close()

    def test_knob_off_is_byte_identical(self, tmp_path):
        cfg = _lif_cfg(exact=False)
        tuner, model = self._tuner(tmp_path, cfg)
        try:
            for p in model.get_perceptrons():
                assert not getattr(p, LIF_HALF_STEP_FLAG, False)
                assert isinstance(p.input_activation, nn.Identity)
                assert not p.activation_scale.requires_grad
                assert not getattr(p, "_mbh_lif_exact_qat", False)
        finally:
            tuner.close()

    def test_goodhart_control_unpaired_config_fails_loud(self, tmp_path):
        cfg = _lif_cfg()
        cfg["lif_per_hop_retiming"] = False
        with pytest.raises(ValueError, match="lif_per_hop_retiming"):
            self._tuner(tmp_path, cfg)


class TestWQFoldOnce:
    def _run_fold_helper(self, model, cfg):
        from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
            WeightQuantizationStep,
        )

        pipeline = MockPipeline(config=cfg)
        fake_step = SimpleNamespace(pipeline=pipeline)
        WeightQuantizationStep._apply_lif_half_step_entry_fold(fake_step, model)

    def _marked_model(self, cfg):
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            manager.update_activation(cfg, p)
        return model

    def test_exact_trained_model_skips_the_wq_entry_fold(self, capsys):
        cfg = _lif_cfg(steps=8)
        cfg["lif_half_step_bias"] = True
        model = self._marked_model(cfg)
        assert apply_lif_half_step_bias_compensation(model, 8) > 0
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        self._run_fold_helper(model, cfg)
        assert "marker-asserted" in capsys.readouterr().out
        for p, b in zip(model.get_perceptrons(), before):
            assert torch.equal(p.layer.bias.detach(), b)

    def test_marker_without_fold_flag_fails_loud(self):
        cfg = _lif_cfg(steps=8)
        cfg["lif_half_step_bias"] = True
        model = self._marked_model(cfg)  # marked but never folded
        with pytest.raises(AssertionError, match="half-step"):
            self._run_fold_helper(model, cfg)

    def test_knob_off_folds_as_shipped(self):
        cfg = _lif_cfg(exact=False)
        cfg["lif_half_step_bias"] = True
        model = make_tiny_supermodel()
        before = [p.layer.bias.detach().clone() for p in model.get_perceptrons()]
        self._run_fold_helper(model, cfg)
        for p, b in zip(model.get_perceptrons(), before):
            if getattr(p, "is_encoding_layer", False):
                assert torch.equal(p.layer.bias.detach(), b)
            else:
                assert getattr(p, LIF_HALF_STEP_FLAG, False)


class TestWQThetaFreeze:
    """[R1/P-L6] the WQ stage trains weights on a FIXED scale lattice: the
    effective-weight fold (per_input_scales x W / theta) and the NAPQ projection
    share theta stamped at WQ entry, so a theta trained under the WQ endpoint
    drifts the lattice out from under the projection (measured: torch<->deployed
    0.9688 vs 1.0000). The exact-QAT arm freezes its in-loop theta at WQ."""

    def _freeze(self, model, cfg):
        from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import (
            WeightQuantizationStep,
        )

        fake_step = SimpleNamespace(pipeline=MockPipeline(config=cfg))
        WeightQuantizationStep._freeze_lif_exact_theta(fake_step, model)

    def test_exact_marked_model_freezes_trainable_theta(self):
        from mimarsinan.spiking.theta_cotrain import promote_theta_for_exact_qat

        cfg = _lif_cfg()
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        model = make_tiny_supermodel(hidden_layers=2)
        promote_theta_for_exact_qat(model)
        for p in model.get_perceptrons():
            manager.update_activation(cfg, p)
        assert any(p.activation_scale.requires_grad for p in model.get_perceptrons())
        self._freeze(model, cfg)
        for p in model.get_perceptrons():
            assert not p.activation_scale.requires_grad

    def test_unmarked_model_is_untouched(self):
        from mimarsinan.spiking.theta_cotrain import promote_activation_scale_per_channel

        cfg = _lif_cfg(exact=False)
        model = make_tiny_supermodel(hidden_layers=2)
        promote_activation_scale_per_channel(model)
        self._freeze(model, cfg)
        assert any(p.activation_scale.requires_grad for p in model.get_perceptrons())


class TestAdaptationPlanReduction:
    def test_arm_reduces_to_finalize_and_verify(self):
        from mimarsinan.tuning.orchestration.lif_adaptation_plan import LifAdaptationPlan

        cfg = _lif_cfg()
        cfg["lif_tanneal"] = True
        cfg["endpoint_recovery_steps"] = 600
        cfg["lif_theta_cotrain"] = True
        plan = LifAdaptationPlan.resolve(cfg)
        assert plan.exact_qat is True
        assert plan.blend_fast_rates == [1.0]
        assert plan.blend_fast_steps_per_rate == 0
        assert plan.tanneal is False
        assert plan.endpoint_recovery_steps == 0
        assert plan.distmatch is False
        assert plan.theta_cotrain is False

    def test_knob_off_plan_is_byte_identical(self):
        from mimarsinan.tuning.orchestration.lif_adaptation_plan import LifAdaptationPlan

        cfg = _lif_cfg(exact=False)
        cfg["lif_tanneal"] = True
        cfg["endpoint_recovery_steps"] = 600
        plan = LifAdaptationPlan.resolve(cfg)
        assert plan.exact_qat is False
        assert plan.blend_fast_rates == [0.25, 0.5, 0.75, 1.0]
        assert plan.tanneal is True
        assert plan.endpoint_recovery_steps == 600
