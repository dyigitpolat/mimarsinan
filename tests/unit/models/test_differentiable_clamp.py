"""Regression tests for DifferentiableClamp.

These cover the failure mode from commit 6a7cb80 ("still flaky"), where the
backward pass used boolean-indexed grad assignment (`grad[below] = ...`) and
could trigger CUDA device-side asserts with large activation tensors (e.g.
produced by ViT). The backward must be broadcast-safe and produce identical
results on CPU and CUDA.
"""

import pytest
import torch

from mimarsinan.models.layers import DifferentiableClamp


_CLAMP_LEAK = 0.01


def _apply(x, a_val, b_val):
    a = torch.tensor(a_val)
    b = torch.tensor(b_val)
    return DifferentiableClamp.apply(x, a, b)


class TestDifferentiableClampForward:
    def test_clamps_to_range(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        out = _apply(x, 0.0, 1.0)
        assert torch.equal(out, torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0]))

    def test_scalar_bounds_accepted(self):
        x = torch.rand(3, 4)
        out = _apply(x, 0.0, 1.0)
        assert out.shape == x.shape

    def test_one_elem_bounds_rejected(self):
        x = torch.tensor([0.5])
        a = torch.tensor([0.0])  # 1-D, shape [1] — not allowed
        b = torch.tensor([1.0])
        with pytest.raises(AssertionError, match="scalar bounds"):
            DifferentiableClamp.apply(x, a, b)


class TestDifferentiableClampBackward:
    def test_grad_inside_range_is_one(self):
        x = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
        out = _apply(x, 0.0, 1.0)
        out.sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_grad_below_range_decays_exponentially(self):
        x = torch.tensor([-1.0], requires_grad=True)
        out = _apply(x, 0.0, 1.0)
        out.sum().backward()
        expected = max(torch.exp(torch.tensor(-1.0)).item(), _CLAMP_LEAK)
        assert x.grad.item() == pytest.approx(expected, rel=1e-5)

    def test_grad_above_range_decays_exponentially(self):
        x = torch.tensor([2.0], requires_grad=True)
        out = _apply(x, 0.0, 1.0)
        out.sum().backward()
        expected = max(torch.exp(torch.tensor(1.0 - 2.0)).item(), _CLAMP_LEAK)
        assert x.grad.item() == pytest.approx(expected, rel=1e-5)

    def test_grad_floored_at_leak(self):
        # Far outside the range, the exponential collapses below _CLAMP_LEAK.
        x = torch.tensor([-100.0], requires_grad=True)
        out = _apply(x, 0.0, 1.0)
        out.sum().backward()
        assert x.grad.item() == pytest.approx(_CLAMP_LEAK, rel=1e-6)


class TestDifferentiableClampShapes:
    """The primary regression target: large, multi-dim inputs must not crash."""

    @pytest.mark.parametrize("shape", [
        (1,), (100,), (32, 768), (4, 196, 768), (2, 3, 32, 32),
    ])
    def test_shape_parametrized(self, shape):
        x = torch.randn(*shape, requires_grad=True)
        out = _apply(x, -0.5, 0.5)
        assert out.shape == torch.Size(shape)
        out.sum().backward()
        assert x.grad.shape == torch.Size(shape)
        assert torch.isfinite(x.grad).all()

    def test_vit_sized_tensor(self):
        # Roughly what ViT produces mid-network on CIFAR-10: batch x patches x hidden.
        x = torch.randn(8, 65, 768, requires_grad=True)
        out = _apply(x, 0.0, 1.0)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
class TestDifferentiableClampCuda:
    """The failure was only on CUDA — parity on CPU matters less than not crashing here."""

    def test_cuda_large_tensor_does_not_crash(self):
        x = torch.randn(4, 196, 768, device="cuda", requires_grad=True)
        out = _apply(x, 0.0, 1.0)
        out.sum().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(x.grad).all()

    def test_cpu_cuda_parity(self):
        x_cpu = torch.randn(2, 64, 128, requires_grad=True)
        out_cpu = _apply(x_cpu, -0.5, 0.5)
        out_cpu.sum().backward()

        x_cuda = x_cpu.detach().cuda().requires_grad_(True)
        out_cuda = _apply(x_cuda, -0.5, 0.5)
        out_cuda.sum().backward()
        torch.cuda.synchronize()

        assert torch.allclose(out_cpu, out_cuda.cpu(), atol=1e-6)
        assert torch.allclose(x_cpu.grad, x_cuda.grad.cpu(), atol=1e-5)
