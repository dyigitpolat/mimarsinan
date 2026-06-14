"""P1: flag-on routing coverage for the non-manager-rate axis families.

The manager-rate family equivalence is in test_axis_routing_equivalence.py; this
covers the blend family (own mechanism) and the thin callable-adapter families
(perceptron-transform / pruning / shift), plus a real-tuner construction check.
"""

import torch
import torch.nn as nn

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    default_config,
)
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import BlendActivation
from mimarsinan.tuning.perceptron_rate import set_blend_rate
from mimarsinan.tuning.axes import (
    BlendAxis,
    LIFAxis,
    TTFSAxis,
    PerceptronTransformAxis,
    NAPQAxis,
    PruningAxis,
    ActivationShiftAxis,
    ActivationAdaptationAxis,
)


def _model_with_blend():
    model = make_tiny_supermodel()
    for p in model.get_perceptrons():
        p.base_activation = BlendActivation(
            p.base_activation, nn.Identity(), 0.0, target_type="X",
        )
    return model


def test_blend_axis_matches_set_blend_rate_and_round_trips():
    cfg = default_config()
    model = _model_with_blend()
    axis = BlendAxis()
    axis.attach(model, None, cfg)

    axis.set_rate(0.4)
    assert all(p.base_activation.rate == 0.4 for p in model.get_perceptrons())

    state = axis.get_extra_state()
    axis.set_rate(0.9)
    axis.set_extra_state(state)
    assert all(p.base_activation.rate == 0.4 for p in model.get_perceptrons())

    # identical to the legacy SSOT
    ref = make_tiny_supermodel()
    for p in ref.get_perceptrons():
        p.base_activation = BlendActivation(p.base_activation, nn.Identity(), 0.0, target_type="X")
    set_blend_rate(ref, 0.4)
    assert [p.base_activation.rate for p in ref.get_perceptrons()] == \
        [p.base_activation.rate for p in model.get_perceptrons()]


def test_blend_subclasses_are_blend_axes():
    assert isinstance(LIFAxis(), BlendAxis)
    assert isinstance(TTFSAxis(), BlendAxis)
    assert LIFAxis().name == "lif" and TTFSAxis().name == "ttfs"


def test_thin_axes_delegate_to_apply_fn():
    calls = []
    PerceptronTransformAxis(lambda r: calls.append(("pt", r))).set_rate(0.3)
    NAPQAxis(lambda r: calls.append(("napq", r))).set_rate(0.6)
    PruningAxis(lambda r: calls.append(("prune", r))).set_rate(0.7)
    ActivationShiftAxis(lambda: calls.append(("shift",))).set_rate(1.0)
    assert calls == [("pt", 0.3), ("napq", 0.6), ("prune", 0.7), ("shift",)]


def test_pruning_axis_recovery_hooks_delegate():
    seen = {}

    def _hooks(r):
        seen["r"] = r
        return ["h"]

    axis = PruningAxis(lambda r: None, recovery_hooks_fn=_hooks)
    assert axis.recovery_hooks(0.5) == ["h"]
    assert seen["r"] == 0.5
    assert PruningAxis(lambda r: None).recovery_hooks(0.5) == []


def test_activation_shift_axis_not_smooth():
    assert ActivationShiftAxis(lambda: None).supports_smooth is False


def test_real_tuner_builds_axis_when_flagged(tmp_path):
    from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
        ActivationAdaptationTuner,
    )

    cfg = default_config()
    cfg["tuning_use_axis"] = True
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    tuner = ActivationAdaptationTuner(pipeline, model, 0.9, 0.001, manager)

    assert isinstance(tuner._axis, ActivationAdaptationAxis)
    tuner._set_rate(0.5)
    assert manager.activation_adaptation_rate == 0.5
    tuner.close()
