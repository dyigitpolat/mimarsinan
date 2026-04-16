"""Regression: ClampTuner's saturation probe on ViT+CUDA must not trip a
CUDA device-side assert.

Covers the forward-only code path that ``test_vit_clamp_backward.py`` does
not reach: ``_probe_clamp_saturation`` runs under ``torch.no_grad()`` and
exercises ``SavedTensorDecorator._maybe_sample`` at the first decorated
activation. Any asynchronous CUDA failure in the preceding forward ops
surfaces at the decorator's first ``.cpu()`` sync and has historically been
misattributed to the sampler itself.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_vit_cifar10_clamp_probe_no_cuda_assert():
    from mimarsinan.models.builders.torch_vit_builder import TorchViTBuilder
    from mimarsinan.models.layers import SavedTensorDecorator
    from mimarsinan.torch_mapping.converter import convert_torch_model

    os.environ["MIMARSINAN_CUDA_DEBUG"] = "1"
    try:
        builder = TorchViTBuilder(
            device="cuda",
            input_shape=(3, 32, 32),
            num_classes=10,
            pipeline_config={},
        )
        model = builder._build_model(num_classes=10).eval()

        flow = convert_torch_model(model, (3, 32, 32), 10, device="cuda")
        flow = flow.to("cuda").eval()

        perceptrons = list(flow.get_perceptrons())
        assert perceptrons, "expected at least one perceptron in the ViT flow"

        decorators = []
        for perceptron in perceptrons:
            decorator = SavedTensorDecorator(sample_to_cpu=True)
            perceptron.activation.decorate(decorator)
            decorators.append(decorator)

        try:
            x = torch.randn(4, 3, 32, 32, device="cuda")
            with torch.no_grad():
                _ = flow(x)
            torch.cuda.synchronize()
        finally:
            for perceptron in perceptrons:
                perceptron.activation.pop_decorator()

        captured = sum(1 for d in decorators if d.latest_output is not None)
        assert captured > 0, "no decorator captured a sampled output tensor"
    finally:
        os.environ.pop("MIMARSINAN_CUDA_DEBUG", None)
