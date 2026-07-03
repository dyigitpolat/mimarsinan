"""Contract of the shared blend-ramp mechanism (``blend_ramp`` + fast-ladder adoption).

The mechanism the KD-blend tuner family (LIF/TTFS) composes over: the linear
blend ramp, the frozen-KD-teacher loss lifecycle, the teacher-distmatch
calibration skeleton, driver adoption, and forward install/uninstall symmetry.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.tuning.forward_install import CascadeForwardInstall
from mimarsinan.tuning.orchestration.blend_ramp import (
    BlendActivation,
    KDClassificationLoss,
    kd_loss_from_config,
    run_teacher_distmatch,
)
from mimarsinan.tuning.orchestration.fast_ladder import FastLadderMixin
from mimarsinan.tuning.orchestration.optimization_driver import OptimizationDriver
from mimarsinan.tuning.teacher import snapshot_frozen_teacher


def _x(n=6, d=5):
    torch.manual_seed(0)
    return torch.randn(n, d)


# ── ramp monotonicity ──────────────────────────────────────────────────────────


class TestBlendRampMonotonicity:
    def _blend(self, rate):
        return BlendActivation(nn.ReLU(), nn.Identity(), rate, target_type="T")

    def test_rate_zero_is_old_exactly(self):
        x = _x()
        torch.testing.assert_close(self._blend(0.0)(x), nn.ReLU()(x), rtol=0, atol=0)

    def test_rate_one_is_target_exactly(self):
        x = _x()
        torch.testing.assert_close(self._blend(1.0)(x), x, rtol=0, atol=0)

    def test_intermediate_is_exact_linear_interpolation(self):
        x = _x()
        expected = 0.7 * nn.ReLU()(x) + 0.3 * x
        torch.testing.assert_close(self._blend(0.3)(x), expected, rtol=0, atol=0)

    def test_output_is_pointwise_monotone_in_rate(self):
        x = _x()
        outs = torch.stack(
            [self._blend(r)(x) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
        )
        diffs = outs[1:] - outs[:-1]
        monotone = (diffs >= 0).all(dim=0) | (diffs <= 0).all(dim=0)
        assert bool(monotone.all())

    def test_activation_type_flips_only_at_full_rate(self):
        blend = BlendActivation(
            nn.ReLU(), nn.Identity(), 0.0, target_type="TGT", old_type="OLD",
        )
        assert blend.activation_type == "OLD"
        blend.rate = 0.999
        assert blend.activation_type == "OLD"
        blend.rate = 1.0
        assert blend.activation_type == "TGT"


# ── KD-teacher lifecycle: freeze / release ─────────────────────────────────────


def _teacher_student():
    torch.manual_seed(1)
    return nn.Linear(5, 3), nn.Linear(5, 3)


class TestKDTeacherLifecycle:
    def test_constructor_freezes_teacher(self):
        teacher, _ = _teacher_student()
        teacher.train()
        KDClassificationLoss(teacher)
        assert not teacher.training
        assert all(not p.requires_grad for p in teacher.parameters())

    def test_backward_trains_student_only(self):
        teacher, student = _teacher_student()
        loss_fn = KDClassificationLoss(teacher)
        x = _x()
        y = torch.randint(0, 3, (x.shape[0],))
        loss_fn(student, x, y).backward()
        assert all(
            p.grad is not None and torch.any(p.grad != 0)
            for p in student.parameters()
        )
        assert all(p.grad is None for p in teacher.parameters())

    def test_alpha_one_is_pure_ce(self):
        teacher, student = _teacher_student()
        loss_fn = KDClassificationLoss(teacher, alpha=1.0)
        x = _x()
        y = torch.randint(0, 3, (x.shape[0],))
        torch.testing.assert_close(
            loss_fn(student, x, y), F.cross_entropy(student(x), y),
        )

    def test_kd_loss_from_config_reads_weighting(self):
        teacher, _ = _teacher_student()
        loss = kd_loss_from_config(
            {"kd_temperature": 5.0, "kd_ce_alpha": 0.7}, teacher,
        )
        assert loss.temperature == pytest.approx(5.0)
        assert loss.alpha == pytest.approx(0.7)

    def test_kd_loss_from_config_defaults_are_kd_heavy(self):
        teacher, _ = _teacher_student()
        loss = kd_loss_from_config({}, teacher)
        assert loss.temperature == pytest.approx(3.0)
        assert loss.alpha == pytest.approx(0.3)

    def test_snapshot_teacher_frozen_and_released_from_student(self):
        _, student = _teacher_student()
        teacher = snapshot_frozen_teacher(student, "cpu")
        assert not teacher.training
        assert all(not p.requires_grad for p in teacher.parameters())
        assert all(p.requires_grad for p in student.parameters())
        x = _x()
        with torch.no_grad():
            before = teacher(x)
            student.weight.add_(1.0)
            after = teacher(x)
        torch.testing.assert_close(before, after, rtol=0, atol=0)


# ── teacher-distmatch calibration skeleton ─────────────────────────────────────


class _Reporter:
    def __init__(self):
        self.reports = []

    def report(self, key, value):
        self.reports.append((key, value))


class _StubTuner:
    def __init__(self):
        self.model = object()
        self._teacher = object()
        self._T = 16
        self.name = "Stub Tuner"
        self.pipeline = SimpleNamespace(reporter=_Reporter())
        self.calibration_calls = []

    def _calibration_inputs(self, n_batches=8):
        self.calibration_calls.append(n_batches)
        return "CAL_X"


class TestRunTeacherDistmatch:
    def test_threads_model_teacher_calx_T_kwargs_and_reports(self):
        tuner = _StubTuner()
        seen = {}

        def matcher(model, teacher, cal_x, T, **kw):
            seen.update(model=model, teacher=teacher, cal_x=cal_x, T=T, kw=kw)
            return {"ok": 1}

        stats = run_teacher_distmatch(
            tuner, matcher, n_batches=3, bias_iters=5, eta=0.5,
        )
        assert stats == {"ok": 1}
        assert tuner.calibration_calls == [3]
        assert seen["model"] is tuner.model
        assert seen["teacher"] is tuner._teacher
        assert seen["cal_x"] == "CAL_X"
        assert seen["T"] == 16
        assert seen["kw"] == {"bias_iters": 5, "eta": 0.5}
        assert tuner.pipeline.reporter.reports == [
            ("Stub Tuner distmatch", {"ok": 1}),
        ]

    def test_default_calibration_batches_is_eight(self):
        tuner = _StubTuner()
        run_teacher_distmatch(tuner, lambda *a, **kw: {})
        assert tuner.calibration_calls == [8]


# ── fast-ladder driver adoption (rate schedule source stays per-tuner) ─────────


class _LadderHost(FastLadderMixin):
    def __init__(self):
        self.pipeline = SimpleNamespace(config={})


def _driver(rates, steps=7, eta=0.2, fast=True):
    return OptimizationDriver(
        fast_ladder=fast,
        fast_ladder_rates=rates,
        fast_ladder_steps_per_rate=steps,
        fast_ladder_eta_min_factor=eta,
    )


class TestAdoptOptimizationDriver:
    def test_adopt_stashes_driver_and_configures_ladder(self):
        host = _LadderHost()
        driver = _driver([0.5, 1.0])
        assert host._adopt_optimization_driver(driver) is driver
        assert host._optimization_driver is driver
        assert host._fixed_ladder_policy is True
        assert host._fixed_ladder_rates == [0.5, 1.0]
        assert host._fast_steps_per_rate == 7
        assert host._fast_eta_min_factor == pytest.approx(0.2)

    def test_ladder_normalized_to_trailing_one(self):
        host = _LadderHost()
        host._adopt_optimization_driver(_driver([0.25, 0.5]))
        assert host._fixed_ladder_rates[-1] == pytest.approx(1.0)

    def test_controller_driver_disables_the_ladder(self):
        host = _LadderHost()
        host._adopt_optimization_driver(_driver([1.0], fast=False))
        assert host._fixed_ladder_policy is False


# ── forward install/uninstall symmetry ─────────────────────────────────────────


class _HostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 3)

    def forward(self, x):
        return self.lin(x)


class _Installer(CascadeForwardInstall):
    def __init__(self, model):
        self.model = model


class TestForwardInstallSymmetry:
    def test_install_then_remove_restores_class_forward(self):
        model = _HostModel()
        x = _x()
        with torch.no_grad():
            baseline = model(x)
        installer = _Installer(model)
        installer._install_forward(lambda t: torch.zeros(t.shape[0], 3))
        assert "forward" in model.__dict__
        with torch.no_grad():
            assert torch.equal(model(x), torch.zeros(x.shape[0], 3))
        installer._remove_forward()
        assert "forward" not in model.__dict__
        with torch.no_grad():
            torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)

    def test_remove_is_idempotent(self):
        model = _HostModel()
        installer = _Installer(model)
        installer._install_forward(lambda t: t)
        installer._remove_forward()
        installer._remove_forward()
        assert "forward" not in model.__dict__

    def test_double_install_asserts(self):
        model = _HostModel()
        installer = _Installer(model)
        installer._install_forward(lambda t: t)
        with pytest.raises(AssertionError):
            installer._install_forward(lambda t: t)


# ── the tuner module keeps re-exporting the mechanism symbols ──────────────────


def test_kd_blend_module_reexports_mechanism_symbols():
    from mimarsinan.tuning.orchestration import kd_blend_adaptation_tuner as m

    assert m.BlendActivation is BlendActivation
    assert m._KDClassificationLoss is KDClassificationLoss
