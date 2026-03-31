"""Unit tests for tolerable instant-drop estimation (pure logic)."""

import pytest

from mimarsinan.tuning.tolerance_calibration import (
    ToleranceCalibrationConfig,
    effective_probe_lr,
    estimate_tolerable_instant_drop,
    initial_tolerance_fn_for_pipeline_if_enabled,
    make_smooth_tolerance_calibration_fn,
    tolerance_config_from_pipeline_config,
)


class TestEstimateTolerableInstantDrop:
    def test_largest_delta_passes_uses_instant_drop(self):
        """8% instant drop at full step, full recovery after one epoch -> tol ~0.08."""
        baseline = 0.9
        cfg = ToleranceCalibrationConfig(
            delta_t_schedule=(1.0, 0.5),
            residual_threshold=1e-3,
            tolerance_min=0.01,
            tolerance_max=0.15,
        )

        def probe(delta_t):
            if delta_t == 1.0:
                return (0.828, 0.9)  # 8% instant, recovered to baseline
            raise AssertionError("should not reach smaller delta")

        tol = estimate_tolerable_instant_drop(baseline, cfg, probe)
        assert tol == pytest.approx((0.9 - 0.828) / 0.9, rel=1e-6)

    def test_falls_back_to_smaller_delta(self):
        baseline = 0.9
        cfg = ToleranceCalibrationConfig(
            delta_t_schedule=(1.0, 0.5),
            residual_threshold=1e-3,
        )

        def probe(delta_t):
            if delta_t == 1.0:
                return (0.5, 0.7)  # poor recovery
            if delta_t == 0.5:
                # residual (0.9 - 0.8999) / 0.9 < 1e-3
                return (0.85, 0.8999)
            raise AssertionError("unexpected delta")

        tol = estimate_tolerable_instant_drop(baseline, cfg, probe)
        assert tol == pytest.approx((0.9 - 0.85) / 0.9, rel=1e-6)

    def test_all_probes_fail_returns_tolerance_min(self):
        cfg = ToleranceCalibrationConfig(
            delta_t_schedule=(1.0, 0.5),
            residual_threshold=1e-3,
            tolerance_min=0.02,
        )

        def probe(_delta_t):
            return (0.1, 0.1)

        assert estimate_tolerable_instant_drop(0.9, cfg, probe) == 0.02

    def test_clamps_to_tolerance_max(self):
        baseline = 0.9
        cfg = ToleranceCalibrationConfig(
            delta_t_schedule=(1.0,),
            residual_threshold=1e-3,
            tolerance_min=0.01,
            tolerance_max=0.15,
        )

        def probe(_delta_t):
            return (0.1, 0.9)  # ~88.9% instant drop

        assert estimate_tolerable_instant_drop(baseline, cfg, probe) == 0.15

    def test_skips_non_positive_delta(self):
        baseline = 0.9
        cfg = ToleranceCalibrationConfig(
            delta_t_schedule=(0.0, 1.0),
            residual_threshold=1e-3,
            tolerance_min=0.03,
        )
        calls = []

        def probe(delta_t):
            calls.append(delta_t)
            return (0.82, 0.9)

        estimate_tolerable_instant_drop(baseline, cfg, probe)
        assert calls == [1.0]

    def test_caps_delta_t_at_one(self):
        baseline = 0.9
        cfg = ToleranceCalibrationConfig(
            delta_t_schedule=(2.0,),
            residual_threshold=1e-3,
        )
        seen = []

        def probe(delta_t):
            seen.append(delta_t)
            return (0.82, 0.9)

        estimate_tolerable_instant_drop(baseline, cfg, probe)
        assert seen == [1.0]


class TestToleranceConfigFromPipeline:
    def test_defaults(self):
        cfg = tolerance_config_from_pipeline_config({})
        assert cfg.delta_t_schedule[0] == 1.0
        assert cfg.residual_threshold == 1e-3
        assert cfg.tolerance_min == 0.01
        assert cfg.tolerance_max == 0.15

    def test_overrides(self):
        cfg = tolerance_config_from_pipeline_config(
            {
                "tuner_smooth_tolerance_delta_schedule": [0.5, 0.25],
                "tuner_smooth_tolerance_residual_threshold": 0.01,
                "tuner_smooth_tolerance_min": 0.02,
                "tuner_smooth_tolerance_max": 0.2,
                "tuner_smooth_tolerance_baseline_epsilon": 1e-6,
            }
        )
        assert cfg.delta_t_schedule == (0.5, 0.25)
        assert cfg.residual_threshold == 0.01
        assert cfg.tolerance_min == 0.02
        assert cfg.tolerance_max == 0.2
        assert cfg.baseline_epsilon == 1e-6


class TestEffectiveProbeLr:
    def test_float_unchanged_with_unit_scale(self):
        assert effective_probe_lr({}, 0.04) == pytest.approx(0.04)

    def test_callable_invoked(self):
        calls = []

        def get_lr():
            calls.append(1)
            return 0.03

        assert effective_probe_lr({}, get_lr) == pytest.approx(0.03)
        assert calls == [1]

    def test_override_wins(self):
        assert effective_probe_lr(
            {"tuner_smooth_tolerance_lr": 0.001},
            0.5,
        ) == pytest.approx(0.001)

    def test_scale_applied(self):
        assert effective_probe_lr(
            {"tuner_smooth_tolerance_lr_scale": 0.5},
            0.1,
        ) == pytest.approx(0.05)

    def test_override_then_scale(self):
        assert effective_probe_lr(
            {
                "tuner_smooth_tolerance_lr": 0.02,
                "tuner_smooth_tolerance_lr_scale": 2.0,
            },
            99.0,
        ) == pytest.approx(0.04)


class TestInitialToleranceFnForPipeline:
    def test_disabled_returns_none(self):
        assert (
            initial_tolerance_fn_for_pipeline_if_enabled(
                {"tuner_calibrate_smooth_tolerance": False},
                clone_state=lambda: 0,
                restore_state=lambda s: None,
                evaluate_at_rate=lambda r: 0.9,
                validate_fn=lambda: 0.9,
                train_validation_epochs=lambda lr, n, w: 0.9,
                lr_probe=0.01,
            )
            is None
        )

    def test_make_calibration_invokes_clone_restore_per_probe(self):
        cfg = {
            "tuner_calibrate_smooth_tolerance": True,
            "tuner_smooth_tolerance_delta_schedule": (1.0,),
            "tuner_smooth_tolerance_residual_threshold": 1e-3,
        }
        restores = []

        def clone_state():
            return 1

        def restore_state(state):
            restores.append(state)

        def evaluate_at_rate(rate):
            return 0.82 if rate >= 1.0 else 0.9

        def validate_fn():
            return 0.9

        train_calls = []

        def train_validation_epochs(lr, n, w):
            train_calls.append((lr, n, w))
            return 0.9

        fn = make_smooth_tolerance_calibration_fn(
            cfg,
            clone_state=clone_state,
            restore_state=restore_state,
            evaluate_at_rate=evaluate_at_rate,
            validate_fn=validate_fn,
            train_validation_epochs=train_validation_epochs,
            lr_probe=0.05,
        )
        tol = fn()
        assert restores == [1]
        assert train_calls == [(0.05, 1, 0)]
        assert tol > 0.01

    def test_callable_lr_probe_resolved_once_per_calibration(self):
        cfg = {
            "tuner_calibrate_smooth_tolerance": True,
            "tuner_smooth_tolerance_delta_schedule": (1.0,),
        }
        lr_calls = []

        def get_lr():
            lr_calls.append(1)
            return 0.07

        fn = make_smooth_tolerance_calibration_fn(
            cfg,
            clone_state=lambda: None,
            restore_state=lambda s: None,
            evaluate_at_rate=lambda r: 0.82,
            validate_fn=lambda: 0.9,
            train_validation_epochs=lambda lr, n, w: 0.9,
            lr_probe=get_lr,
        )
        fn()
        assert lr_calls == [1]
