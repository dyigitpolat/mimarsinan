"""LIFAdaptationTuner symmetric patch/unpatch on ``model.forward``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
    _CycleAccurateForward,
    LIFAdaptationTuner,
)


class _FakePerceptronModel(nn.Module):
    """Minimal model exposing ``get_perceptrons`` for the tuner."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def get_perceptrons(self):
        return []

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover (unused here)
        return self.fc(x)


def test_install_cycle_accurate_forward_assert_blocks_double_patch() -> None:
    model = _FakePerceptronModel()
    model.forward = _CycleAccurateForward(model, T=4)
    with pytest.raises(AssertionError, match="already patched"):
        # Second install on top of an existing instance-level forward must fail.
        _install_check(model, T=4)


def _install_check(model: nn.Module, T: int) -> None:
    """Mirror the install assertion path."""
    assert "forward" not in model.__dict__, (
        "LIFAdaptationTuner: model.forward is already patched; double-install would "
        "shadow the prior wrapper. Call _after_run on the previous tuner first."
    )
    model.forward = _CycleAccurateForward(model, T=int(T))


def test_unpatch_removes_instance_forward() -> None:
    """After symmetric unpatch, ``model.forward`` falls back to the class method."""
    model = _FakePerceptronModel()
    bound_class_forward = type(model).forward
    _install_check(model, T=4)
    assert "forward" in model.__dict__
    # Unpatch (mirrors _after_run cleanup).
    del model.forward
    assert "forward" not in model.__dict__
    # Bound method retrieval falls back to the class method.
    assert model.forward.__func__ is bound_class_forward


def test_unpatch_idempotent() -> None:
    """Calling unpatch twice does not raise — the second `del` is guarded."""
    model = _FakePerceptronModel()
    _install_check(model, T=4)
    try:
        del model.forward
    except AttributeError:
        pass
    # Second del is the idempotent case.
    try:
        del model.forward
    except AttributeError:
        pass  # expected; not an error
    assert "forward" not in model.__dict__


def test_tuner_install_assert_blocks_double_patch() -> None:
    """The real tuner's ``_install_cycle_accurate_forward`` should assert when
    the model already has an instance-level forward."""
    class _Tuner:
        def __init__(self, model, T):
            self.model = model
            self._T = int(T)
            self._patched_forward = False

        def _install_cycle_accurate_forward(self):
            from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
                LIFAdaptationTuner,
            )
            # Reuse the actual method
            LIFAdaptationTuner._install_cycle_accurate_forward(self)

    model = _FakePerceptronModel()
    tuner = _Tuner(model, T=4)
    tuner._install_cycle_accurate_forward()
    with pytest.raises(AssertionError, match="already patched"):
        tuner._install_cycle_accurate_forward()
    # Cleanup
    del model.forward


def test_tuner_after_run_unpatches_even_when_cycle_accurate() -> None:
    """``_after_run`` must remove ``model.forward`` instance attr regardless of
    the cycle-accurate flag. Prior versions left the patch in place when
    ``self._cycle_accurate == True``, leaking into downstream pipeline stages."""
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    model = _FakePerceptronModel()

    class _StubTuner:
        def __init__(self):
            self.model = model
            self._cycle_accurate = True
            self._patched_forward = True
            self._continue_called = False
            self._set_rate_called = False
            self.adaptation_manager = type("M", (), {"lif_active": False})()
            self.pipeline = type("P", (), {"config": {"firing_mode": "Default"}})()
            self._final_metric = None
            self._committed_rate = None

        def _continue_to_full_rate(self):
            self._continue_called = True

        def _set_rate(self, r):
            self._set_rate_called = True

        def _ensure_pipeline_threshold(self):
            return 1.0

    model.forward = _CycleAccurateForward(model, T=4)
    stub = _StubTuner()
    LIFAdaptationTuner._after_run(stub)
    assert "forward" not in model.__dict__, (
        "after_run should always unpatch model.forward (was only when not cycle_accurate)"
    )
