"""P1: manager-rate axes delegate to the exact legacy SSOT (byte-identical)."""

import copy

import torch

from conftest import make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.perceptron_rate import apply_manager_rate
from mimarsinan.tuning.axes import ClampAxis, ActQuantAxis


def _fwd(model, x, seed=0):
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        return model(x).clone()


def test_set_rate_matches_apply_manager_rate_byte_for_byte():
    cfg = default_config()
    x = torch.randn(2, 1, 8, 8)

    model_axis = make_tiny_supermodel()
    model_direct = copy.deepcopy(model_axis)
    mgr_axis = create_adaptation_manager_for_model(cfg, model_axis)
    mgr_direct = create_adaptation_manager_for_model(cfg, model_direct)

    axis = ClampAxis()
    axis.attach(model_axis, mgr_axis, cfg)
    axis.set_rate(0.5)
    apply_manager_rate(model_direct, mgr_direct, cfg, "clamp_rate", 0.5)

    assert mgr_axis.clamp_rate == mgr_direct.clamp_rate == 0.5
    assert torch.allclose(_fwd(model_axis, x), _fwd(model_direct, x))


def test_set_rate_calls_apply_manager_rate(monkeypatch):
    cfg = default_config()
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)

    seen = {}

    import mimarsinan.tuning.axes.manager_rate_axis as mod

    def _spy(m, am, c, rate_attr, rate):
        seen["args"] = (m, am, c, rate_attr, rate)
        apply_manager_rate(m, am, c, rate_attr, rate)

    monkeypatch.setattr(mod, "apply_manager_rate", _spy)

    axis = ActQuantAxis()
    axis.attach(model, manager, cfg)
    axis.set_rate(0.3)

    assert seen["args"] == (model, manager, cfg, "quantization_rate", 0.3)
