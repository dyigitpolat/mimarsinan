"""Tests for ``tuning_target_floor_on_real_target`` (default-off, byte-identical).

The death-spiral bug: ``run()`` overwrites the adaptation target/original/floor
with ``baseline_val`` — the rate-0 read, which on an ANN→LIF collapse reads
*collapsed* (~0.27 on the live ResNet-50 job). The relaxation then drives the
target down to ``baseline_val * (1 - tol)`` where it sticks, so the tuner stops
aiming to recover toward the real ANN target (~0.9 on the representative split /
~0.72 official).

When the flag is ON and the REAL pipeline target (``pipeline.get_target_metric()``,
already passed to the constructor as ``target_accuracy``) is above the collapsed
baseline, ``run()`` must KEEP the real target as the relaxation anchor and cap the
relaxation floor BELOW the real target but never down to ``baseline_val * 0.9`` —
so the model aims to recover toward the ANN.

Flag OFF must replay the legacy baseline-anchored block byte-identically.
"""

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster


class _RunTuner(SmoothAdaptationTuner):
    """A tuner whose rate-0 baseline calibration reads a SCRIPTED collapsed value,
    so ``run()``'s baseline block runs against a known ``baseline_val`` while the
    pipeline's real target (``get_target_metric``) stays high. The driver loop is
    short-circuited so the test only exercises the baseline-anchoring block."""

    def __init__(self, pipeline, model, target_accuracy, lr, *, baseline_val):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._scripted_baseline = float(baseline_val)
        self.trainer.validate_n_batches = lambda n: self._scripted_baseline
        self.trainer.validate = lambda: self._scripted_baseline
        self.trainer.train_steps_until_target = lambda *a, **k: None
        self.trainer.test = lambda: self._scripted_baseline

    def _update_and_evaluate(self, rate):
        return self._scripted_baseline

    def _find_lr(self):
        return 0.001

    # Short-circuit the scheduler loop: ``run()``'s baseline block is what we
    # measure, not the full ramp.
    def _run_with_scheduler(self):
        return self._scripted_baseline


def _pipeline(tmp_path, *, real_target, degradation_tolerance=0.1, flag=False):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = degradation_tolerance
    cfg["tuning_target_floor_on_real_target"] = flag
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline.set_target_metric(real_target)
    return pipeline


def _make_tuner(pipeline, *, real_target, baseline_val):
    return _RunTuner(
        pipeline, make_tiny_supermodel(), target_accuracy=real_target,
        lr=0.001, baseline_val=baseline_val,
    )


class TestFlagOffByteIdentical:
    """Default (flag off): ``run()`` anchors target/original/floor on the
    collapsed ``baseline_val`` — the legacy death-spiral behavior, unchanged."""

    def test_run_anchors_on_collapsed_baseline(self, tmp_path):
        real_target, baseline_val, tol = 0.9, 0.27, 0.1
        pipeline = _pipeline(tmp_path, real_target=real_target,
                             degradation_tolerance=tol, flag=False)
        tuner = _make_tuner(pipeline, real_target=real_target, baseline_val=baseline_val)
        tuner.run()

        adj = tuner.target_adjuster
        assert adj.target_metric == pytest.approx(baseline_val)
        assert adj.original_metric == pytest.approx(baseline_val)
        assert adj.floor == pytest.approx(baseline_val * (1.0 - tol))

    def test_legacy_relaxation_floors_at_collapsed_baseline(self, tmp_path):
        """With the flag off, the relaxation death-spiral bottoms out at
        ``baseline_val * (1 - tol)`` (~0.243) — the collapsed floor, far below
        the real target (the bug this fix addresses)."""
        real_target, baseline_val, tol = 0.9, 0.27, 0.1
        pipeline = _pipeline(tmp_path, real_target=real_target,
                             degradation_tolerance=tol, flag=False)
        tuner = _make_tuner(pipeline, real_target=real_target, baseline_val=baseline_val)
        tuner.run()

        adj = tuner.target_adjuster
        for _ in range(200):
            adj.update_target(0.0)
        # Death-spiral: floors at the COLLAPSED baseline's floor, NOT the real target.
        assert adj.get_target() == pytest.approx(baseline_val * (1.0 - tol))
        assert adj.get_target() < real_target * (1.0 - tol)


class TestFlagOnAnchorsOnRealTarget:
    """Flag on, real target above collapsed baseline: keep the real target as the
    relaxation anchor and cap the floor BELOW it but ABOVE the collapsed floor."""

    def test_original_metric_stays_real_target(self, tmp_path):
        real_target, baseline_val, tol = 0.9, 0.27, 0.1
        pipeline = _pipeline(tmp_path, real_target=real_target,
                             degradation_tolerance=tol, flag=True)
        tuner = _make_tuner(pipeline, real_target=real_target, baseline_val=baseline_val)
        tuner.run()

        adj = tuner.target_adjuster
        # The relaxation anchor (original_metric) is the REAL target, not baseline.
        assert adj.original_metric == pytest.approx(real_target)
        assert adj.target_metric == pytest.approx(real_target)

    def test_floor_between_baseline_and_real_target(self, tmp_path):
        real_target, baseline_val, tol = 0.9, 0.27, 0.1
        pipeline = _pipeline(tmp_path, real_target=real_target,
                             degradation_tolerance=tol, flag=True)
        tuner = _make_tuner(pipeline, real_target=real_target, baseline_val=baseline_val)
        tuner.run()

        adj = tuner.target_adjuster
        # Floor capped strictly below the real target, at/above the achievable baseline.
        assert adj.floor >= baseline_val - 1e-9
        assert adj.floor < real_target
        assert adj.floor == pytest.approx(real_target * (1.0 - tol))

    def test_relaxation_cannot_collapse_to_baseline(self, tmp_path):
        """200 missed-target relaxations cannot drive the target down to the
        collapsed baseline floor; it stays anchored near the real target."""
        real_target, baseline_val, tol = 0.9, 0.27, 0.1
        pipeline = _pipeline(tmp_path, real_target=real_target,
                             degradation_tolerance=tol, flag=True)
        tuner = _make_tuner(pipeline, real_target=real_target, baseline_val=baseline_val)
        tuner.run()

        adj = tuner.target_adjuster
        for _ in range(200):
            adj.update_target(0.0)
        relaxed = adj.get_target()
        # Aims to recover toward the ANN: floored near the real target, NOT ~0.243.
        assert relaxed == pytest.approx(real_target * (1.0 - tol))
        assert relaxed > baseline_val
        assert relaxed >= real_target * (1.0 - tol) - 1e-9


class TestFlagOnDegenerateFallsBackToBaseline:
    """Flag on but the real target is at/below the collapsed baseline (degenerate /
    no upstream ANN target): fall back to baseline anchoring — no regression."""

    def test_no_real_target_falls_back(self, tmp_path):
        baseline_val, tol = 0.27, 0.1
        # get_target_metric == 0.0 (MockPipeline default) => no valid real target.
        pipeline = _pipeline(tmp_path, real_target=0.0,
                             degradation_tolerance=tol, flag=True)
        tuner = _make_tuner(pipeline, real_target=0.0, baseline_val=baseline_val)
        tuner.run()

        adj = tuner.target_adjuster
        assert adj.target_metric == pytest.approx(baseline_val)
        assert adj.original_metric == pytest.approx(baseline_val)
        assert adj.floor == pytest.approx(baseline_val * (1.0 - tol))

    def test_real_target_below_baseline_falls_back(self, tmp_path):
        real_target, baseline_val, tol = 0.20, 0.27, 0.1
        pipeline = _pipeline(tmp_path, real_target=real_target,
                             degradation_tolerance=tol, flag=True)
        tuner = _make_tuner(pipeline, real_target=real_target, baseline_val=baseline_val)
        tuner.run()

        adj = tuner.target_adjuster
        # Real target below the achievable baseline => baseline anchoring stands.
        assert adj.original_metric == pytest.approx(baseline_val)
        assert adj.floor == pytest.approx(baseline_val * (1.0 - tol))


class TestAdjusterRelaxationArithmetic:
    """Unit-level proof of the relaxation cap, independent of ``run()``: an
    adjuster anchored on the real target with a real-target floor never decays
    below that floor, whereas the legacy baseline-anchored adjuster collapses."""

    def test_real_target_anchored_adjuster_holds_floor(self):
        real_target, tol = 0.9, 0.1
        adj = AdaptationTargetAdjuster(real_target, decay=0.95,
                                       floor_ratio=1.0 - tol)
        for _ in range(500):
            adj.update_target(0.0)
        assert adj.get_target() == pytest.approx(real_target * (1.0 - tol))

    def test_baseline_anchored_adjuster_collapses(self):
        baseline_val, tol = 0.27, 0.1
        adj = AdaptationTargetAdjuster(baseline_val, decay=0.95,
                                       floor_ratio=1.0 - tol)
        for _ in range(500):
            adj.update_target(0.0)
        # The collapsed floor the death-spiral sticks at.
        assert adj.get_target() == pytest.approx(baseline_val * (1.0 - tol))
