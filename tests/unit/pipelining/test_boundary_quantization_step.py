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

    def test_one_quantizer_per_entry_on_the_live_scale(self):
        p = _perceptron()
        (q,) = install_boundary_quantizers([p], activation_bits=8)
        assert isinstance(q, ValueGridQuantizer)
        assert q.scale is p.input_activation_scale
        assert [m for m in p.modules() if isinstance(m, ValueGridQuantizer)] == [q]


class TestCalibration:
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
        assert float(first.input_activation_scale) == float(batch.abs().max())
        assert abs(
            float(second.input_activation_scale) - float(hidden.abs().max())
        ) < 1e-6

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
        assert float(p.input_activation_scale) == float(batch.abs().max())

    def test_maxima_accumulate_over_batches(self):
        p = _perceptron(3, 3)
        (q,) = install_boundary_quantizers([p], 8)
        small, large = torch.full((2, 3), 0.5), torch.full((2, 3), 2.0)
        calibrate_boundary_scales(p, [p], [q], [small, large], quantile=1.0)
        assert float(p.input_activation_scale) == 2.0

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
        assert abs(float(p.input_activation_scale) - MIN_BOUNDARY_SCALE) < 1e-9

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
        scale = p.input_activation_scale.detach().clone()
        with torch.no_grad():
            got = p(x)
            want = p.activation(p.layer(quantize_to_value_grid(x, scale, 8)))
        torch.testing.assert_close(got, want)

    def test_scale_write_is_live_for_the_quantizer(self):
        # One-writer currency: a later scale write moves the grid without
        # re-installation.
        p = _perceptron()
        (q,) = install_boundary_quantizers([p], 8)
        calibrate_boundary_scales(p, [p], [q], [torch.randn(4, 4)])
        p.input_activation_scale.data.fill_(4.0)
        x = torch.tensor([[3.9, -3.9, 0.02, 1.0]])
        with torch.no_grad():
            got = p(x)
            want = p.activation(
                p.layer(quantize_to_value_grid(x, torch.tensor(4.0), 8))
            )
        torch.testing.assert_close(got, want)
