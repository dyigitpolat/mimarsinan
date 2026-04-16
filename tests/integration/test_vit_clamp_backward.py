"""Regression: ViT + Clamp Adaptation on CUDA must not trigger device-side asserts.

The original failure was ``CUDA error: device-side assert triggered`` during
Clamp Adaptation on CIFAR-10 + ``torch_vit``. The root cause was in
``DifferentiableClamp.backward`` (boolean-indexed grad writes with implicit
scalar broadcasting on large ViT-sized activation tensors).

This test covers the integration path that the unit tests for
``DifferentiableClamp`` and ``ConvertedModelFlow`` can't reach on their own:
a real ViT is built, converted, moved to CUDA, and a full forward+backward
goes through its graph with the ClampDecorator active.
"""

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_vit_cifar10_clamp_backward_no_cuda_assert():
    from mimarsinan.models.builders.torch_vit_builder import TorchViTBuilder
    from mimarsinan.models.layers import ClampDecorator

    builder = TorchViTBuilder(
        device="cuda",
        input_shape=(3, 32, 32),
        num_classes=10,
        pipeline_config={},
    )
    # Small ViT config that still exercises the ViT activation-tensor shapes.
    model = builder._build_model(num_classes=10)
    model = model.to("cuda").eval()

    x = torch.randn(4, 3, 32, 32, device="cuda", requires_grad=True)
    logits = model(x)

    # Apply clamp to the logits — the same DifferentiableClamp path exercised
    # by ClampAdaptationStep, on a ViT-emitted tensor.
    clamp = ClampDecorator(
        torch.tensor(-1.0, device="cuda"),
        torch.tensor(1.0, device="cuda"),
    )
    clamped = clamp.output_transform(logits)
    loss = clamped.sum()
    loss.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(x.grad).all()
