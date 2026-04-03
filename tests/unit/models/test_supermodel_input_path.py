"""Tests for PerceptronFlow input path: single input activation (no InputCQ)."""

import torch

from conftest import TinyPerceptronFlow, default_config
from mimarsinan.models.layers import LeakyGradReLU
from mimarsinan.tuning.adaptation_manager import AdaptationManager


def _make_flow(input_shape=(1, 28, 28), num_classes=10):
    flow = TinyPerceptronFlow(input_shape=input_shape, num_classes=num_classes)
    am = AdaptationManager()
    cfg = default_config()
    cfg["input_shape"] = input_shape
    cfg["num_classes"] = num_classes
    for p in flow.get_perceptrons():
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        am.update_activation(cfg, p)
    return flow


def test_input_activation_identity_forward():
    """Default input activation is identity; forward runs without extra clamp preprocessor."""
    flow = _make_flow()
    flow.eval()
    x = torch.rand(2, 1, 28, 28)
    ia = flow.get_input_activation()
    with torch.no_grad():
        out = flow(x)
    assert out.shape == (2, 10)
    assert torch.allclose(ia(x), x)
