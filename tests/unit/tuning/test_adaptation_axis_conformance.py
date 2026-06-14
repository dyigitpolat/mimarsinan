"""Spec §14.1 conformance for AdaptationAxis adapters (P1, manager-rate family)."""

import copy

import pytest
import torch

from conftest import make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.axes import (
    AdaptationAxis,
    ClampAxis,
    ActQuantAxis,
    NoiseAxis,
    ActivationAdaptationAxis,
)

_AXES = [ClampAxis, ActQuantAxis, NoiseAxis, ActivationAdaptationAxis]


def _model_and_manager():
    cfg = default_config()
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return model, manager, cfg


def _fwd(model, x, seed=0):
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        return model(x).clone()


@pytest.fixture
def x():
    return torch.randn(2, 1, 8, 8)


@pytest.mark.parametrize("AxisCls", _AXES)
def test_axis_implements_protocol(AxisCls):
    assert isinstance(AxisCls(), AdaptationAxis)


@pytest.mark.parametrize("AxisCls", _AXES)
def test_attach_is_idempotent(AxisCls, x):
    model, manager, cfg = _model_and_manager()
    axis = AxisCls()
    axis.attach(model, manager, cfg)
    axis.attach(model, manager, cfg)  # second attach must not change behavior
    axis.set_rate(0.5)
    once = _fwd(model, x)

    model2, manager2, cfg2 = _model_and_manager()
    model2.load_state_dict(model.state_dict())
    axis2 = AxisCls()
    axis2.attach(model2, manager2, cfg2)
    axis2.set_rate(0.5)
    assert torch.allclose(once, _fwd(model2, x))


@pytest.mark.parametrize("AxisCls", _AXES)
def test_set_rate_zero_is_identity_and_reversible(AxisCls, x):
    model, manager, cfg = _model_and_manager()
    axis = AxisCls()
    axis.attach(model, manager, cfg)
    base = _fwd(model, x)            # manager starts at rate 0 → identity
    axis.set_rate(1.0)
    axis.set_rate(0.0)
    assert torch.allclose(base, _fwd(model, x), atol=1e-6)


@pytest.mark.parametrize("AxisCls", _AXES)
def test_extra_state_round_trip(AxisCls, x):
    model, manager, cfg = _model_and_manager()
    axis = AxisCls()
    axis.attach(model, manager, cfg)

    axis.set_rate(0.6)
    state = axis.get_extra_state()
    assert float(state) == pytest.approx(0.6)
    at_06 = _fwd(model, x)

    axis.set_rate(0.2)
    axis.set_extra_state(state)
    assert float(getattr(manager, axis.rate_attr)) == pytest.approx(0.6)
    assert torch.allclose(at_06, _fwd(model, x))


@pytest.mark.parametrize("AxisCls", _AXES)
def test_descriptor_is_stable(AxisCls):
    model, manager, cfg = _model_and_manager()
    a, b = AxisCls(), AxisCls()
    assert a.descriptor() == b.descriptor()
    a.attach(model, manager, cfg)
    before = a.descriptor()
    a.set_rate(0.5)
    assert a.descriptor() == before


@pytest.mark.parametrize("AxisCls", _AXES)
def test_rate_grid_runs(AxisCls):
    model, manager, cfg = _model_and_manager()
    axis = AxisCls()
    axis.attach(model, manager, cfg)
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        axis.set_rate(r)
        assert float(getattr(manager, axis.rate_attr)) == pytest.approx(r)


@pytest.mark.parametrize("AxisCls", _AXES)
def test_defaults_are_noops(AxisCls):
    axis = AxisCls()
    assert list(axis.tunable_parameters()) == []
    assert axis.recovery_hooks(0.5) == []
    assert axis.calibrate(None, None) is None
    assert axis.set_decision_seed(0) is None
