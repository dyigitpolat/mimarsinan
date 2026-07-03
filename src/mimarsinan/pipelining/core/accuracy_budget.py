"""Cross-step cumulative accuracy-drop budget for the deployment pipeline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class AccuracyBudget:
    """Tracks cumulative test-accuracy drop across a pipeline run.

    ``budget_total <= 0`` disables cross-step enforcement (step_floor returns
    only the per-step value).
    """

    budget_total: float = 0.05
    reference: Optional[float] = None
    latest: Optional[float] = None

    def seeded(self) -> bool:
        """True once the first non-zero test metric has been observed."""
        return self.reference is not None

    def observe(self, metric: float) -> None:
        """Record a fresh test metric; zero/negative values are ignored for seeding."""
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
        """Lowest absolute test accuracy allowed by the cross-step budget (``None`` before seeding)."""
        if self.reference is None or self.budget_total <= 0.0:
            return None
        return self.reference - self.budget_total

    def step_floor(self, previous_metric: float, per_step_tolerance: float) -> float:
        """Hard floor for the next step: max of the per-step and cross-step floors (``0.0`` before seeding)."""
        if self.reference is None:
            return 0.0
        per_step = max(0.0, float(previous_metric)) * max(0.0, 1.0 - float(per_step_tolerance))
        cross = self.absolute_floor()
        if cross is None:
            return per_step
        return max(per_step, cross)

    def warn_if_over_budget(self, step_name: str) -> None:
        """Emit a warning when the cumulative drop exceeds the budget (per-step assertion stays the hard stop)."""
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
