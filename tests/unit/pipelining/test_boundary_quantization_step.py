"""[mvm AQ] Boundary Quantization step: applicability, calibration, installation."""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.value_quantizer import (
    ValueGridQuantizer,
    quantize_to_value_grid,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    MIN_BOUNDARY_SCALE,
    BoundaryQuantizationStep,
    calibrate_boundary_scales,
    install_boundary_quantizers,
)


def _perceptron(out_ch=4, in_f=4, seed=0):
    torch.manual_seed(seed)
    return Perceptron(out_ch, in_f)


class TestAppliesTo:
    def test_armed_mvm_plan(self):
        plan = DeploymentPlan.resolve(
            {"core_semantics": "mvm", "activation_quantization": True}
        )
        assert BoundaryQuantizationStep.applies_to(plan) is True

    def test_unarmed_mvm_plan(self):
        plan = DeploymentPlan.resolve({"core_semantics": "mvm"})
        assert BoundaryQuantizationStep.applies_to(plan) is False

    def test_spiking_plan_never_applies(self):
        plan = DeploymentPlan.resolve(
            {"spiking_mode": "lif", "activation_quantization": True}
        )
        assert BoundaryQuantizationStep.applies_to(plan) is False


class TestInstallation:
    def test_quantizer_is_inert_until_calibrated(self):
        p = _perceptron()
        x = torch.randn(5, 4)
        with torch.no_grad():
            want = p(x)
        install_boundary_quantizers([p], activation_bits=8)
        with torch.no_grad():
            got = p(x)
        torch.testing.assert_close(got, want)

    def test_the_grid_is_owned_by_the_quantizer_not_the_perceptron(self):
        # The event domain's input_activation_scale keeps its single wire
        # meaning; the AQ grid is a typed field on the quantizer's buffer.
        p = _perceptron()
        before = float(p.input_activation_scale)
        (q,) = install_boundary_quantizers([p], activation_bits=8)
        assert isinstance(q, ValueGridQuantizer)
        assert "scale" in dict(q.named_buffers())
        assert q.grid.armed is False
        q.calibrate(2.0)
        assert q.grid.scale == 2.0 and q.grid.bits == 8 and q.grid.armed
        assert float(p.input_activation_scale) == before  # untouched


class TestCalibration:
    def test_grid_step_is_one_lsb(self):
        (q,) = install_boundary_quantizers([_perceptron()], activation_bits=8)
        q.calibrate(2.54)
        assert q.grid.levels == 127
        assert abs(q.grid.step - 2.54 / 127) < 1e-7  # fp32 buffer

    def test_scales_are_input_quantiles_at_the_seam(self):
        first, second = _perceptron(4, 4, seed=1), _perceptron(2, 4, seed=2)
        model = nn.Sequential(first, second)
        batch = torch.randn(64, 4)
        with torch.no_grad():
            hidden = first(batch)  # pre-install == post-install (inert) input
        quantizers = install_boundary_quantizers([first, second], 8)
        calibrate_boundary_scales(
            model, [first, second], quantizers, [batch], quantile=1.0
        )
        assert quantizers[0].grid.scale == float(batch.abs().max())
        assert abs(quantizers[1].grid.scale - float(hidden.abs().max())) < 1e-6

    def test_functional_invocation_path_is_calibrated(self):
        # The conv mappers invoke ``input_activation`` directly (no
        # Perceptron.__call__); the seam hook must still see the boundary.
        p = _perceptron()
        (q,) = install_boundary_quantizers([p], 8)

        class FunctionalMapper(nn.Module):
            def forward(self, x):
                x = p.input_activation(x)
                return p.layer(x)

        batch = torch.randn(32, 4)
        calibrate_boundary_scales(
            FunctionalMapper(), [p], [q], [batch], quantile=1.0
        )
        assert q.grid.scale == float(batch.abs().max())

    def test_maxima_accumulate_over_batches(self):
        p = _perceptron(3, 3)
        (q,) = install_boundary_quantizers([p], 8)
        small, large = torch.full((2, 3), 0.5), torch.full((2, 3), 2.0)
        calibrate_boundary_scales(p, [p], [q], [small, large], quantile=1.0)
        assert q.grid.scale == 2.0

    def test_unreached_entry_fails_loud(self):
        reached, dead = _perceptron(seed=3), _perceptron(seed=4)
        quantizers = install_boundary_quantizers([reached, dead], 8)
        with pytest.raises(RuntimeError, match="never reached"):
            calibrate_boundary_scales(
                reached, [reached, dead], quantizers, [torch.randn(4, 4)]
            )

    def test_tiny_maxima_floor_at_min_scale(self):
        p = _perceptron()
        (q,) = install_boundary_quantizers([p], 8)
        calibrate_boundary_scales(
            p, [p], [q], [torch.full((2, 4), 1e-7)], quantile=1.0
        )
        assert abs(q.grid.scale - MIN_BOUNDARY_SCALE) < 1e-9

    def test_hooks_are_removed_after_calibration(self):
        p = _perceptron()
        (q,) = install_boundary_quantizers([p], 8)
        calibrate_boundary_scales(p, [p], [q], [torch.randn(2, 4)])
        assert not q._forward_pre_hooks


class TestQuantizedForward:
    def test_forward_snaps_input_to_the_grid(self):
        p = _perceptron()
        (q,) = install_boundary_quantizers([p], 8)
        x = torch.randn(5, 4)
        calibrate_boundary_scales(p, [p], [q], [x], quantile=1.0)
        scale = q.scale.detach().clone()
        with torch.no_grad():
            got = p(x)
            want = p.activation(p.layer(quantize_to_value_grid(x, scale, 8)))
        torch.testing.assert_close(got, want)

    def test_recalibration_moves_the_grid_without_reinstallation(self):
        p = _perceptron()
        (q,) = install_boundary_quantizers([p], 8)
        calibrate_boundary_scales(p, [p], [q], [torch.randn(4, 4)])
        q.calibrate(4.0)
        x = torch.tensor([[3.9, -3.9, 0.02, 1.0]])
        with torch.no_grad():
            got = p(x)
            want = p.activation(
                p.layer(quantize_to_value_grid(x, torch.tensor(4.0), 8))
            )
        torch.testing.assert_close(got, want)
