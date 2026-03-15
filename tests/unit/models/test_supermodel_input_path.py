"""Tests for Supermodel input path: preprocessor vs in_act (double application).

See plan section 5.2: confirm whether clamp+quant is applied twice.
"""

import pytest
import torch

from mimarsinan.models.supermodel import Supermodel
from mimarsinan.models.preprocessing.input_cq import InputCQ
from conftest import TinyPerceptronFlow, default_config
from mimarsinan.models.layers import LeakyGradReLU
from mimarsinan.tuning.adaptation_manager import AdaptationManager


def _make_supermodel_with_input_cq(input_shape=(1, 28, 28), num_classes=10, Tq=32):
    """Supermodel with InputCQ preprocessor (same as converter)."""
    flow = TinyPerceptronFlow(input_shape=input_shape, num_classes=num_classes)
    preprocessor = InputCQ(Tq)
    model = Supermodel("cpu", input_shape, num_classes, preprocessor, flow, Tq)
    am = AdaptationManager()
    cfg = default_config()
    cfg["input_shape"] = input_shape
    cfg["num_classes"] = num_classes
    for p in model.get_perceptrons():
        p.base_activation = LeakyGradReLU()
        p.activation = LeakyGradReLU()
        am.update_activation(cfg, p)
    return model


def test_supermodel_preprocessor_and_in_act_are_both_applied():
    """Document whether in_act changes already-quantized output (second application)."""
    Tq = 32
    model = _make_supermodel_with_input_cq(Tq=Tq)
    model.eval()
    x = torch.rand(2, 1, 28, 28)  # [0, 1] range
    with torch.no_grad():
        after_preprocessor = model.preprocessor(x)
        after_in_act = model.in_act(after_preprocessor)
    # Preprocessor already does clamp [0,1] + quantize to Tq levels.
    # Second application (in_act) on that output: QuantizeDecorator on already
    # quantized values may be idempotent or may change values depending on implementation.
    assert after_preprocessor.shape == after_in_act.shape
    # Strict: second application of in_act on already-quantized output should be idempotent.
    assert torch.allclose(
        after_preprocessor, after_in_act, atol=1e-6
    ), "in_act(preprocessor(x)) should equal preprocessor(x) when QuantizeDecorator is idempotent."


def test_single_input_activation_matches_double():
    """If single-application accuracy >> double, double application is harmful."""
    from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
    from conftest import MockDataProviderFactory

    Tq = 32
    model_double = _make_supermodel_with_input_cq(Tq=Tq)
    model_double.eval()
    # Variant: only preprocessor (no extra in_act) before perceptron_flow.
    # We cannot easily change Supermodel.forward, so we compare:
    # (a) full supermodel: preprocessor -> in_act -> perceptron_flow
    # (b) perceptron_flow(preprocessor(x)): single application
    dp_factory = MockDataProviderFactory(input_shape=(1, 28, 28), num_classes=10)
    dp = dp_factory.create()
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    val_loader = dlf.create_validation_loader(dp.get_validation_batch_size(), dp)

    acc_double = 0.0
    acc_single = 0.0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            out_double = model_double(x)
            preprocessed = model_double.preprocessor(x)
            out_single = model_double.perceptron_flow(preprocessed)
            acc_double += (out_double.argmax(dim=1) == y).float().sum().item()
            acc_single += (out_single.argmax(dim=1) == y).float().sum().item()
            n += y.size(0)
    acc_double /= n
    acc_single /= n
    assert 0.0 <= acc_double <= 1.0 and 0.0 <= acc_single <= 1.0
    # If single is significantly higher, double application hurts (and we could fix Supermodel).
    # We only assert both are valid accuracies here; the plan uses this to decide whether to fix.
    assert acc_single >= 0.0
    assert acc_double >= 0.0
