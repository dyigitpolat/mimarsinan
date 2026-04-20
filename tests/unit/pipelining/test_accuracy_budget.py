"""Tests for the cross-step accuracy-drop budget.

AccuracyBudget lives on the pipeline and tracks the total accuracy drop
across steps relative to the first non-zero test metric (the baseline,
typically set by Pretraining / Weight Preloading).

Rules:
- ``seeded()`` is ``False`` before any non-zero metric is observed.
- When the first non-zero metric is observed, it becomes ``reference``.
- ``consumed`` is ``max(0.0, reference - current)``.
- ``remaining`` is ``budget_total - consumed`` (floored at 0.0).
- ``step_floor(previous_metric, per_step_tolerance)`` returns
  ``max(previous_metric * (1 - per_step_tolerance), reference - budget_total)``
  once seeded; before then it returns ``previous_metric * (1 - per_step_tolerance)``.
"""

import pytest

from mimarsinan.pipelining.accuracy_budget import AccuracyBudget


class TestAccuracyBudgetSeeding:
    def test_unseeded_on_zero(self):
        budget = AccuracyBudget(budget_total=0.10)
        budget.observe(0.0)
        assert not budget.seeded()
        assert budget.reference is None

    def test_seeded_on_first_positive(self):
        budget = AccuracyBudget(budget_total=0.10)
        budget.observe(0.85)
        assert budget.seeded()
        assert budget.reference == pytest.approx(0.85)

    def test_reference_does_not_change_after_seeding(self):
        budget = AccuracyBudget(budget_total=0.10)
        budget.observe(0.85)
        budget.observe(0.90)  # Later higher metric — not a new reference.
        assert budget.reference == pytest.approx(0.85)


class TestAccuracyBudgetConsumption:
    def test_consumed_zero_when_metric_matches_reference(self):
        budget = AccuracyBudget(budget_total=0.10)
        budget.observe(0.85)
        assert budget.consumed() == pytest.approx(0.0)
        assert budget.remaining() == pytest.approx(0.10)

    def test_consumed_reflects_drop(self):
        budget = AccuracyBudget(budget_total=0.10)
        budget.observe(0.85)
        budget.observe(0.80)
        assert budget.consumed() == pytest.approx(0.05)
        assert budget.remaining() == pytest.approx(0.05)

    def test_remaining_floored_at_zero(self):
        budget = AccuracyBudget(budget_total=0.10)
        budget.observe(0.85)
        budget.observe(0.50)
        assert budget.remaining() == pytest.approx(0.0)

    def test_improvement_does_not_give_back_budget_beyond_zero(self):
        budget = AccuracyBudget(budget_total=0.10)
        budget.observe(0.85)
        budget.observe(0.90)
        assert budget.consumed() == pytest.approx(0.0)
        assert budget.remaining() == pytest.approx(0.10)


class TestAccuracyBudgetStepFloor:
    def test_step_floor_before_seeding_uses_only_per_step(self):
        budget = AccuracyBudget(budget_total=0.10)
        assert budget.step_floor(previous_metric=0.80, per_step_tolerance=0.05) == pytest.approx(0.0)
        assert budget.step_floor(previous_metric=0.0, per_step_tolerance=0.05) == pytest.approx(0.0)

    def test_step_floor_after_seeding_is_max_of_both(self):
        budget = AccuracyBudget(budget_total=0.10)
        budget.observe(0.90)
        # Per-step: previous=0.80, tolerance=0.05 -> 0.80 * 0.95 = 0.76
        # Cross-step: reference - budget_total = 0.90 - 0.10 = 0.80
        # max = 0.80
        floor = budget.step_floor(previous_metric=0.80, per_step_tolerance=0.05)
        assert floor == pytest.approx(0.80)

    def test_step_floor_per_step_dominates_when_larger(self):
        budget = AccuracyBudget(budget_total=0.30)
        budget.observe(0.90)
        # Per-step: 0.80 * 0.95 = 0.76
        # Cross-step: 0.90 - 0.30 = 0.60
        # max = 0.76
        floor = budget.step_floor(previous_metric=0.80, per_step_tolerance=0.05)
        assert floor == pytest.approx(0.76)

    def test_step_floor_cross_step_dominates_when_larger(self):
        budget = AccuracyBudget(budget_total=0.02)
        budget.observe(0.90)
        # Per-step: 0.80 * 0.95 = 0.76
        # Cross-step: 0.90 - 0.02 = 0.88
        # max = 0.88
        floor = budget.step_floor(previous_metric=0.80, per_step_tolerance=0.05)
        assert floor == pytest.approx(0.88)
