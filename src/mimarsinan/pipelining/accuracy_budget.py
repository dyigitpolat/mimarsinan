"""Cross-step accuracy-drop budget for the deployment pipeline.

The pipeline's step-level assertion is
``current_test >= previous_test * tolerance``, which prevents a *single*
step from dropping accuracy by more than ``degradation_tolerance``. On
its own, this lets many adjacent in-tolerance drops accumulate silently.

:class:`AccuracyBudget` bounds the **total** drop across the pipeline
relative to the first non-zero test metric (the "reference" — typically
set by Pretraining / Weight Preloading). Early steps that have not yet
produced a real test metric see ``seeded() is False`` and get no
additional constraint from the budget; the per-step formula still runs
unchanged.

The budget is deliberately *soft*: it returns a ``step_floor`` that the
tuner can use as a hard floor, and the pipeline's assertion is the final
gate. If an individual step runs clean under the per-step formula but
causes the cross-step budget to go negative, a warning is emitted via
:meth:`warn_if_over_budget`.

See also: ``pipeline.py`` (sets target metric after each step) and
``unified_tuner.py`` (uses ``pipeline.accuracy_budget.step_floor(...)``
as the hard-floor passed to ``_attempt_recovery_if_below_floor``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class AccuracyBudget:
    """Tracks cumulative test-accuracy drop across a pipeline run.

    :param budget_total: Absolute maximum drop allowed across the entire
        pipeline (e.g. 0.05 for "no more than 5 percentage points total").
        A ``budget_total <= 0`` disables cross-step enforcement (the
        :meth:`step_floor` returns only the per-step value).
    """

    budget_total: float = 0.05
    reference: Optional[float] = None
    latest: Optional[float] = None

    def seeded(self) -> bool:
        """True once the first non-zero test metric has been observed."""
        return self.reference is not None

    def observe(self, metric: float) -> None:
        """Record a fresh test metric.

        Zero / negative values are ignored for seeding (the pipeline uses
        ``0.0`` as a placeholder before any step has produced a real
        accuracy measurement).
        """
        value = float(metric)
        if value > 0.0 and self.reference is None:
            self.reference = value
        if value > 0.0:
            self.latest = value

    def consumed(self) -> float:
        """Total drop consumed since the reference, floored at 0."""
        if self.reference is None or self.latest is None:
            return 0.0
        return max(0.0, self.reference - self.latest)

    def remaining(self) -> float:
        """Budget remaining (floored at 0)."""
        if self.reference is None:
            return max(0.0, self.budget_total)
        return max(0.0, self.budget_total - self.consumed())

    def absolute_floor(self) -> Optional[float]:
        """Lowest absolute test accuracy allowed by the cross-step budget.

        Returns ``None`` before the budget is seeded — before then no
        cross-step constraint applies.
        """
        if self.reference is None or self.budget_total <= 0.0:
            return None
        return self.reference - self.budget_total

    def step_floor(self, previous_metric: float, per_step_tolerance: float) -> float:
        """Hard floor for the next step.

        Before seeding: returns ``0.0`` (no cross-step constraint yet —
        callers apply their own per-step guard). After seeding: the max
        of the per-step floor ``previous * (1 - per_step_tolerance)`` and
        the cross-step floor ``reference - budget_total``.

        ``previous_metric`` may be ``0.0`` (pipeline's placeholder before
        the first real metric); then the per-step component is ``0.0``
        too.
        """
        if self.reference is None:
            return 0.0
        per_step = max(0.0, float(previous_metric)) * max(0.0, 1.0 - float(per_step_tolerance))
        cross = self.absolute_floor()
        if cross is None:
            return per_step
        return max(per_step, cross)

    def warn_if_over_budget(self, step_name: str) -> None:
        """Emit a warning when the cumulative drop exceeds the budget.

        The pipeline's per-step assertion remains the hard stop; this
        warning surfaces a cross-step regression that passed the per-step
        gate but nibbled through the global budget.
        """
        if self.reference is None:
            return
        if self.consumed() > self.budget_total:
            warnings.warn(
                f"[{step_name}] cumulative accuracy drop "
                f"{self.consumed():.4f} exceeds cross-step budget "
                f"{self.budget_total:.4f} "
                f"(reference={self.reference:.4f}, latest={self.latest:.4f})",
                stacklevel=2,
            )
