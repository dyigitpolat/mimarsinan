"""[mvm AQ] value-grid quantization math and the STE module."""

import torch

from mimarsinan.models.nn.activations.value_quantizer import (
    ValueGridQuantizer,
    quantize_to_value_grid,
    value_grid_levels,
)


class TestGridMath:
    def test_levels(self):
        assert value_grid_levels(8) == 127
        assert value_grid_levels(4) == 7

    def test_grid_points_are_exact(self):
        scale = torch.tensor(1.0)
        x = torch.tensor([0.0, 1.0, -1.0, 0.5])
        q = quantize_to_value_grid(x, scale, 8)
        step = 1.0 / 127
        torch.testing.assert_close(q / step, torch.round(q / step))
        assert q[1].item() == 1.0 and q[2].item() == -1.0

    def test_saturates_at_scale(self):
        q = quantize_to_value_grid(torch.tensor([5.0, -5.0]), torch.tensor(1.0), 8)
        torch.testing.assert_close(q, torch.tensor([1.0, -1.0]))

    def test_zero_scale_is_identity(self):
        x = torch.randn(4)
        torch.testing.assert_close(quantize_to_value_grid(x, torch.tensor(0.0), 8), x)

    def test_quantization_error_bounded_by_half_step(self):
        x = torch.rand(1000) * 2 - 1
        q = quantize_to_value_grid(x, torch.tensor(1.0), 8)
        assert (q - x).abs().max().item() <= (1.0 / 127) / 2 + 1e-9


class TestModule:
    def test_ste_gradient_passes_through(self):
        x = torch.randn(8, requires_grad=True)
        module = ValueGridQuantizer(8, scale=1.0)
        module(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_calibration_arms_the_owned_grid(self):
        module = ValueGridQuantizer(8)
        x = torch.randn(4)
        torch.testing.assert_close(module(x), x)  # inert until calibrated
        module.calibrate(1.0)
        assert not torch.equal(module(torch.tensor([0.003])), torch.tensor([0.003]))

    def test_scale_is_a_buffer_that_travels_with_the_module(self):
        # Declared state: it survives state_dict / .to() / deepcopy without
        # aliasing another module's parameter.
        import copy

        module = ValueGridQuantizer(8)
        module.calibrate(2.5)
        assert "scale" in module.state_dict()
        assert not list(module.parameters())
        assert copy.deepcopy(module).grid == module.grid
