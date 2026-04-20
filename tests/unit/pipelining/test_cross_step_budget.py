"""Phase A2: cross-step accuracy budget enforcement.

Introduces ``pipeline.baseline_test_metric`` + ``pipeline.global_floor``.
The per-step assertion in ``Pipeline._run_step`` must check both:

  new >= max(previous * tolerance, pipeline.global_floor)

so cumulative-but-individually-small drops cannot ratchet past the user's
total degradation budget. Early non-training steps that report 0.0 must
not falsely anchor the baseline at zero; once a step sets a real baseline
(via ``pipeline.set_baseline_test_metric``) the global floor follows it.
"""

from __future__ import annotations

import pytest

from mimarsinan.pipelining.pipeline import Pipeline
from mimarsinan.pipelining.pipeline_step import PipelineStep


class _FixedMetricStep(PipelineStep):
    """Simple step that reports a fixed pipeline_metric()."""

    def __init__(self, pipeline, metric):
        super().__init__([], [], [], [], pipeline)
        self._metric = metric

    def process(self):
        pass

    def validate(self):
        return self._metric

    def pipeline_metric(self):
        return self._metric


class _BaselineSeedingStep(_FixedMetricStep):
    """Like _FixedMetricStep but also sets the pipeline's test baseline."""

    def process(self):
        self.pipeline.set_baseline_test_metric(self._metric)


class TestBaselineAndGlobalFloor:
    def test_baseline_setter_seeds_global_floor(self, tmp_path):
        p = Pipeline(str(tmp_path / "cache"))
        p.tolerance = 0.95
        p.degradation_tolerance = 0.05

        assert p.baseline_test_metric is None
        assert p.global_floor == 0.0

        p.set_baseline_test_metric(0.80)

        assert p.baseline_test_metric == pytest.approx(0.80)
        assert p.global_floor == pytest.approx(0.80 * (1 - 0.05))

    def test_baseline_setter_monotonic_non_decreasing(self, tmp_path):
        """The baseline is anchored to the highest observed value -- later
        assertions must not be able to ratchet the floor down by reporting
        degraded metrics."""
        p = Pipeline(str(tmp_path / "cache"))
        p.degradation_tolerance = 0.05
        p.set_baseline_test_metric(0.80)
        p.set_baseline_test_metric(0.70)  # should be ignored
        assert p.baseline_test_metric == pytest.approx(0.80)

    def test_global_floor_zero_until_baseline_set(self, tmp_path):
        """Early steps propagate 0.0 until a real baseline is published; the
        per-step assertion must tolerate that explicitly (new >= 0.0)."""
        p = Pipeline(str(tmp_path / "cache"))
        p.degradation_tolerance = 0.05
        # Two 0.0-metric steps are allowed because no baseline is set yet.
        p.add_pipeline_step("s1", _FixedMetricStep(p, 0.0))
        p.add_pipeline_step("s2", _FixedMetricStep(p, 0.0))
        p.run()  # must not raise


class TestCrossStepAccumulation:
    def test_small_drops_within_total_budget_pass(self, tmp_path):
        """Per-step drops within tolerance AND within the cumulative budget:
        must all pass."""
        p = Pipeline(str(tmp_path / "cache"))
        p.tolerance = 0.95
        p.degradation_tolerance = 0.05

        p.add_pipeline_step("seed", _BaselineSeedingStep(p, 0.80))
        p.add_pipeline_step("s1", _FixedMetricStep(p, 0.79))
        p.add_pipeline_step("s2", _FixedMetricStep(p, 0.78))
        p.add_pipeline_step("s3", _FixedMetricStep(p, 0.77))
        p.add_pipeline_step("s4", _FixedMetricStep(p, 0.76))
        p.run()  # 0.76 >= 0.80 * 0.95 = 0.76 → exactly at floor, OK

    def test_large_single_drop_exceeds_global_floor(self, tmp_path):
        """A per-step drop that is within tolerance relative to the previous
        step's metric but exceeds the global floor must still fail."""
        p = Pipeline(str(tmp_path / "cache"))
        # Per-step tolerance 0.90 (allow 10% per step) but global budget 5%.
        p.tolerance = 0.90
        p.degradation_tolerance = 0.05

        p.add_pipeline_step("seed", _BaselineSeedingStep(p, 0.80))
        # 0.80 * 0.90 = 0.72 (per-step OK at 0.73)
        # but global floor = 0.80 * 0.95 = 0.76 (fails)
        p.add_pipeline_step("big_drop", _FixedMetricStep(p, 0.73))

        with pytest.raises(AssertionError, match="global floor|global_floor|floor"):
            p.run()

    def test_cumulative_small_drops_exceed_global_floor(self, tmp_path):
        """Multiple individually-small drops that cumulatively exceed the
        global floor must be rejected."""
        p = Pipeline(str(tmp_path / "cache"))
        p.tolerance = 0.98  # per-step permissive
        p.degradation_tolerance = 0.05  # but total ratchet is 5%

        p.add_pipeline_step("seed", _BaselineSeedingStep(p, 0.80))
        p.add_pipeline_step("s1", _FixedMetricStep(p, 0.79))
        p.add_pipeline_step("s2", _FixedMetricStep(p, 0.78))
        p.add_pipeline_step("s3", _FixedMetricStep(p, 0.77))
        # Still inside per-step tolerance (0.77 * 0.98 = 0.755), but global
        # floor is 0.80 * 0.95 = 0.76, so the next drop trips it.
        p.add_pipeline_step("s4", _FixedMetricStep(p, 0.755))

        with pytest.raises(AssertionError):
            p.run()


class _SkippedStep(_FixedMetricStep):
    """Pipeline step that opts out of the floor check.

    Mirrors how pass-through / setup / preparation steps behave: they
    legitimately don't produce a meaningful test metric and should not
    wipe ``previous_metric`` to 0, nor should they be asserted against
    the global floor themselves.
    """

    skip_from_floor_check = True


class TestSkipListForZeroMetricSteps:
    def test_skipped_step_does_not_reset_previous_metric(self, tmp_path):
        """After a baseline is established, a skipped step reporting 0.0
        must not reset the running ``previous_metric`` chain -- the next
        real step must still be compared to the prior nonzero metric."""
        p = Pipeline(str(tmp_path / "cache"))
        p.tolerance = 0.95
        p.degradation_tolerance = 0.05

        p.add_pipeline_step("seed", _BaselineSeedingStep(p, 0.80))
        p.add_pipeline_step("passthrough", _SkippedStep(p, 0.0))
        # Without the skip-list fix, "passthrough" would set target_metric to 0
        # and then 0.78 > 0 * 0.95 = 0 would trivially pass -- silently allowing
        # any future drop.  With the skip-list honoured, previous_metric stays
        # at 0.80 and 0.78 is compared against max(0.80*0.95, global_floor)=0.76.
        p.add_pipeline_step("real", _FixedMetricStep(p, 0.78))
        p.run()
        assert p.get_target_metric() == pytest.approx(0.78)

    def test_skipped_step_bypass_catches_drop_from_retained_baseline(self, tmp_path):
        """If a skipped step is followed by a real step that would have
        passed under the buggy 'previous_metric = 0' behaviour but fails
        against the real previous metric, the assertion MUST still fire."""
        p = Pipeline(str(tmp_path / "cache"))
        p.tolerance = 0.95
        p.degradation_tolerance = 0.20  # loosen global floor so per-step catches

        p.add_pipeline_step("seed", _BaselineSeedingStep(p, 0.80))
        p.add_pipeline_step("passthrough", _SkippedStep(p, 0.0))
        # 0.70 / 0.80 = 0.875 < 0.95 per-step tolerance: must fail.
        p.add_pipeline_step("drop", _FixedMetricStep(p, 0.70))

        with pytest.raises(AssertionError):
            p.run()

    def test_skipped_step_itself_not_asserted_against_floor(self, tmp_path):
        """A skipped step reporting 0.0 must not trip either floor check
        -- it's opted out of assertions entirely."""
        p = Pipeline(str(tmp_path / "cache"))
        p.tolerance = 0.95
        p.degradation_tolerance = 0.05

        p.add_pipeline_step("seed", _BaselineSeedingStep(p, 0.80))
        # Without skip-list, this would fail: 0.0 < max(0.80*0.95, 0.80*0.95)=0.76
        p.add_pipeline_step("passthrough", _SkippedStep(p, 0.0))
        p.run()
