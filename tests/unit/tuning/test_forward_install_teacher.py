"""SSOT primitives: lazy-executor forward install + frozen-teacher snapshot."""

from __future__ import annotations

import pickle

import torch
import torch.nn as nn

from mimarsinan.tuning.forward_install import CascadeForwardInstall, LazyExecutorForward
from mimarsinan.tuning.teacher import freeze_module, snapshot_frozen_teacher


class _CountingForward(LazyExecutorForward):
    """Builds its executor once, lazily; counts builds to prove caching."""

    builds = 0

    def _build_executor(self):
        type(self).builds += 1
        return lambda x: x * 2

    def _run(self, x):
        return self._ensure_executor(self._build_executor)(x)


class TestLazyExecutorForward:
    def test_executor_built_once_and_cached(self):
        _CountingForward.builds = 0
        fwd = _CountingForward(model=nn.Identity(), T=4)
        x = torch.ones(3)
        assert torch.equal(fwd(x), x * 2)
        assert torch.equal(fwd(x), x * 2)
        assert _CountingForward.builds == 1, "executor must be built exactly once"

    def test_executor_dropped_on_pickle(self):
        fwd = _CountingForward(model=nn.Identity(), T=4)
        fwd(torch.ones(2))  # build it
        assert fwd._executor is not None
        restored = pickle.loads(pickle.dumps(fwd))
        assert restored._executor is None, "executor must not survive pickling"
        assert restored.T == 4

    def test_run_not_implemented_on_base(self):
        import pytest

        base = LazyExecutorForward(model=nn.Identity(), T=4)
        with pytest.raises(NotImplementedError):
            base(torch.ones(1))


class _Owner(CascadeForwardInstall):
    def __init__(self, model):
        self.model = model


class TestCascadeForwardInstall:
    def test_install_then_remove_restores_class_forward(self):
        model = nn.Linear(3, 3)
        owner = _Owner(model)
        sentinel = lambda x: x  # noqa: E731
        owner._install_forward(sentinel)
        assert model.__dict__["forward"] is sentinel
        owner._remove_forward()
        assert "forward" not in model.__dict__

    def test_double_install_asserts(self):
        import pytest

        model = nn.Linear(3, 3)
        owner = _Owner(model)
        owner._install_forward(lambda x: x)
        with pytest.raises(AssertionError, match="already patched"):
            owner._install_forward(lambda x: x)

    def test_remove_is_idempotent(self):
        owner = _Owner(nn.Linear(3, 3))
        owner._remove_forward()  # never installed -> no-op
        owner._install_forward(lambda x: x)
        owner._remove_forward()
        owner._remove_forward()  # second remove -> no-op


class TestSnapshotFrozenTeacher:
    def test_teacher_is_frozen_eval_and_independent(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        model.train()
        teacher = snapshot_frozen_teacher(model, device="cpu")

        assert not teacher.training, "teacher must be in eval mode"
        assert all(not p.requires_grad for p in teacher.parameters())
        # Independent copy: mutating the source must not change the teacher.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        s_param = next(model.parameters())
        t_param = next(teacher.parameters())
        assert not torch.equal(s_param, t_param)
        # Source is restored to its device and left trainable.
        assert all(p.requires_grad for p in model.parameters())

    def test_freeze_module_returns_same_instance(self):
        m = nn.Linear(2, 2)
        out = freeze_module(m)
        assert out is m and not m.training
        assert all(not p.requires_grad for p in m.parameters())
