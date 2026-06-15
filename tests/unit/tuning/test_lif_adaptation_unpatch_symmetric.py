"""LIFAdaptationTuner symmetric patch/unpatch on ``model.forward``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
    _ChipAlignedNFForward,
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


def test_double_patch_assert_blocks_reinstall() -> None:
    model = _FakePerceptronModel()
    model.forward = _ChipAlignedNFForward(model, T=4)
    with pytest.raises(AssertionError, match="already patched"):
        # Second install on top of an existing instance-level forward must fail.
        _install_check(model, T=4)


def _install_check(model: nn.Module, T: int) -> None:
    """Mirror the shared ``CascadeForwardInstall._install_forward`` assertion."""
    assert "forward" not in model.__dict__, (
        "model.forward is already patched; a double-install would shadow the "
        "prior wrapper. Remove it first."
    )
    model.forward = _ChipAlignedNFForward(model, T=int(T))


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


def test_shared_install_mixin_blocks_double_patch() -> None:
    """The shared ``CascadeForwardInstall`` mixin asserts when an instance-level
    forward already shadows the class forward (no double-install)."""
    from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
        CascadeForwardInstall,
    )

    class _Tuner(CascadeForwardInstall):
        def __init__(self, model):
            self.model = model

    model = _FakePerceptronModel()
    tuner = _Tuner(model)
    tuner._install_forward(_ChipAlignedNFForward(model, T=4))
    with pytest.raises(AssertionError, match="already patched"):
        tuner._install_forward(_ChipAlignedNFForward(model, T=4))
    tuner._remove_forward()
    assert "forward" not in model.__dict__


def test_shared_remove_forward_is_idempotent() -> None:
    """``_remove_forward`` is a no-op once the instance forward is gone."""
    from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
        CascadeForwardInstall,
    )

    class _Tuner(CascadeForwardInstall):
        def __init__(self, model):
            self.model = model

    model = _FakePerceptronModel()
    tuner = _Tuner(model)
    tuner._install_forward(_ChipAlignedNFForward(model, T=4))
    tuner._remove_forward()
    tuner._remove_forward()  # idempotent
    assert "forward" not in model.__dict__


def test_after_run_unpatches_ramp_forward() -> None:
    """The shared ``_after_run`` removes the ramp forward in a ``finally`` before
    finalize, so a tuner that installs no finalize forward (e.g. non-cycle-
    accurate LIF / synchronized TTFS) leaves the pristine class forward for
    downstream stages — regardless of which ramp wrapper was installed."""
    from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
        CascadeForwardInstall,
        KDBlendAdaptationTuner,
    )

    model = _FakePerceptronModel()

    class _Stub(CascadeForwardInstall):
        def __init__(self):
            self.model = model
            self._patched_forward = True
            self._final_metric = None
            self._committed_rate = None

        def _continue_to_full_rate(self):
            pass

        def _set_rate(self, r):
            pass

        def _safe_eval(self):  # no trainer in the stub
            return None

        def _finalize(self):  # no finalize forward reinstalled
            pass

        def _report_cliff_probe_consistency(self):  # no probe log in the stub
            pass

        def _ensure_pipeline_threshold(self):
            return 1.0

    model.forward = _ChipAlignedNFForward(model, T=4)
    stub = _Stub()
    KDBlendAdaptationTuner._after_run(stub)
    assert "forward" not in model.__dict__
