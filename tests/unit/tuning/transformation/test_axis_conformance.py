"""Full axis-conformance suite (report IV.4 / spec §14.1).

Extends ``tests/unit/tuning/test_adaptation_axis_conformance.py`` (manager-rate
family only) to cover ``BlendAxis`` as well, and emphasizes the state-carriage
contract (``get_extra_state`` / ``set_extra_state`` round-trip is *behavioral*,
not just numeric). Each axis is wrapped in a small ``_AxisCase`` adapter so the
single parametrized contract runs uniformly over heterogeneous attach signatures
(manager-rate axes drive ``adaptation_manager.<rate_attr>``; blend axes drive the
live per-perceptron ``BlendActivation.rate``).

The spec §14.1 properties asserted for every axis:
- ``set_rate(0)`` is identity (rate-0 forward == the pre-attach forward) and
  reversible (any rate followed by ``set_rate(0)`` returns to identity).
- ``attach`` is idempotent (a second attach does not perturb behavior).
- ``descriptor()`` is stable across instances and across ``set_rate`` calls.
- ``get_extra_state`` / ``set_extra_state`` round-trips both the carried value
  and the resulting forward output.
- ``set_decision_seed`` is a no-op on the deterministic axes here.
"""

import pytest
import torch
import torch.nn as nn

from conftest import make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import BlendActivation
from mimarsinan.tuning.axes import (
    AdaptationAxis,
    ClampAxis,
    ActQuantAxis,
    NoiseAxis,
    ActivationAdaptationAxis,
    BlendAxis,
)


# ---------------------------------------------------------------------------
# Per-axis test adapters — uniform surface over heterogeneous attach signatures
# ---------------------------------------------------------------------------

class _AxisCase:
    """A built (model, manager, axis) triple + the introspection a test needs."""

    def __init__(self, axis, model, manager, *, read_rate):
        self.axis = axis
        self.model = model
        self.manager = manager
        self._read_rate = read_rate

    def read_rate(self):
        """The rate the axis currently carries, read from where it lives."""
        return self._read_rate(self)


def _manager_case(AxisCls):
    cfg = default_config()
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    axis = AxisCls()
    axis.attach(model, manager, cfg)

    def read(case):
        return float(getattr(case.manager, case.axis.rate_attr))

    return _AxisCase(axis, model, manager, read_rate=read)


def _blend_case():
    cfg = default_config()
    model = make_tiny_supermodel()
    for p in model.get_perceptrons():
        p.base_activation = BlendActivation(
            p.base_activation, nn.Identity(), 0.0, target_type="X",
        )
    axis = BlendAxis()
    axis.attach(model, None, cfg)

    def read(case):
        rates = [p.base_activation.rate for p in case.model.get_perceptrons()]
        assert all(r == rates[0] for r in rates), "blend rate diverged across perceptrons"
        return float(rates[0])

    return _AxisCase(axis, model, None, read_rate=read)


# name -> (factory, AxisCls-or-None). Manager-rate family + the blend family.
_CASES = {
    "clamp": (lambda: _manager_case(ClampAxis), ClampAxis),
    "act_quant": (lambda: _manager_case(ActQuantAxis), ActQuantAxis),
    "noise": (lambda: _manager_case(NoiseAxis), NoiseAxis),
    "act_adapt": (lambda: _manager_case(ActivationAdaptationAxis), ActivationAdaptationAxis),
    "blend": (_blend_case, BlendAxis),
}

_CASE_IDS = list(_CASES)


@pytest.fixture(params=_CASE_IDS)
def case(request):
    factory, _cls = _CASES[request.param]
    return factory()


@pytest.fixture
def x():
    return torch.randn(2, 1, 8, 8)


def _fwd(model, x, seed=0):
    # Seed before each forward so stochastic-mask axes (ActQuantAxis, NoiseAxis)
    # are reproducible: a fixed rate must yield a fixed forward across calls.
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        return model(x).clone()


# ---------------------------------------------------------------------------
# Spec §14.1 contract — runs over every axis in _CASES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _CASE_IDS)
def test_axis_implements_protocol(name):
    _factory, cls = _CASES[name]
    assert isinstance(cls(), AdaptationAxis)


def test_set_rate_zero_is_identity_and_reversible(case, x):
    base = _fwd(case.model, x)        # every axis starts at rate 0 -> identity
    case.axis.set_rate(1.0)
    case.axis.set_rate(0.0)
    assert case.read_rate() == pytest.approx(0.0)
    assert torch.allclose(base, _fwd(case.model, x), atol=1e-6)


def test_attach_is_idempotent(case, x):
    # Re-attaching the SAME context must not perturb subsequent behavior.
    if case.manager is not None:
        case.axis.attach(case.model, case.manager, case.axis._config)
    else:
        case.axis.attach(case.model, None, case.axis._config)
    case.axis.set_rate(0.5)
    once = _fwd(case.model, x)

    case.axis.set_rate(0.0)
    if case.manager is not None:
        case.axis.attach(case.model, case.manager, case.axis._config)
    else:
        case.axis.attach(case.model, None, case.axis._config)
    case.axis.set_rate(0.5)
    assert torch.allclose(once, _fwd(case.model, x), atol=1e-6)


def test_descriptor_is_stable():
    # Per-axis (not over the parametrized live case, which mutates state): two
    # fresh instances agree, and descriptor survives attach + set_rate.
    for nm in _CASE_IDS:
        _factory, cls = _CASES[nm]
        a, b = cls(), cls()
        assert a.descriptor() == b.descriptor()
        case = _CASES[nm][0]()
        before = case.axis.descriptor()
        case.axis.set_rate(0.5)
        assert case.axis.descriptor() == before


def test_extra_state_round_trips_value_and_forward(case, x):
    case.axis.set_rate(0.6)
    assert case.read_rate() == pytest.approx(0.6)
    state = case.axis.get_extra_state()
    at_06 = _fwd(case.model, x)

    case.axis.set_rate(0.2)
    assert case.read_rate() == pytest.approx(0.2)

    case.axis.set_extra_state(state)
    assert case.read_rate() == pytest.approx(0.6)        # carried value restored
    assert torch.allclose(at_06, _fwd(case.model, x), atol=1e-6)  # forward restored


def test_extra_state_is_decoupled_from_live_state(case):
    """A captured state must not track later rate changes (it is a snapshot)."""
    case.axis.set_rate(0.3)
    snap = case.axis.get_extra_state()
    snap_copy = list(snap) if isinstance(snap, list) else snap
    case.axis.set_rate(0.9)
    if isinstance(snap, list):
        assert snap == snap_copy, "list state aliased and mutated under set_rate"
    else:
        assert snap == snap_copy, "scalar state changed under set_rate"
    case.axis.set_extra_state(snap)
    assert case.read_rate() == pytest.approx(0.3)


def test_set_decision_seed_is_noop(case, x):
    case.axis.set_rate(0.5)
    before = _fwd(case.model, x)
    assert case.axis.set_decision_seed(0) is None
    assert case.read_rate() == pytest.approx(0.5)
    assert torch.allclose(before, _fwd(case.model, x), atol=1e-6)


def test_rate_grid_is_reflected(case):
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        case.axis.set_rate(r)
        assert case.read_rate() == pytest.approx(r)


def test_defaults_are_noops(case):
    # The orchestration-facing defaults the driver/recovery services rely on.
    assert list(case.axis.tunable_parameters()) == []
    assert case.axis.recovery_hooks(0.5) == []
    assert case.axis.finalize(case.model) is None


# ---------------------------------------------------------------------------
# Blend-specific state-carriage emphasis (per-perceptron list state)
# ---------------------------------------------------------------------------

def test_blend_state_is_per_perceptron_list():
    case = _blend_case()
    case.axis.set_rate(0.4)
    state = case.axis.get_extra_state()
    n = len(list(case.model.get_perceptrons()))
    assert isinstance(state, list) and len(state) == n
    assert all(r == pytest.approx(0.4) for r in state)


def test_blend_set_extra_state_restores_each_perceptron():
    case = _blend_case()
    case.axis.set_rate(0.7)
    state = case.axis.get_extra_state()
    # perturb to distinct per-perceptron rates, then restore from the snapshot
    for i, p in enumerate(case.model.get_perceptrons()):
        p.base_activation.rate = 0.1 * (i + 1)
    case.axis.set_extra_state(state)
    assert all(
        p.base_activation.rate == pytest.approx(0.7)
        for p in case.model.get_perceptrons()
    )
