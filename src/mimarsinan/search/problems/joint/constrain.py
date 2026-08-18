"""The DECLARED-constraint channel of the joint architecture + hardware search.

A constraint shapes a region the optimizer can SEE; it is never an exception a
run discovers after it has picked a winner. This channel is also one of the
seams the TS1 accountant charges, so it names itself when it asks.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from mimarsinan.mapping.verification.onchip_fraction import (
    estimate_onchip_fraction,
)
from mimarsinan.search.constraints import ConstraintReport, onchip_floor_violation

from .types import (
    CONSTRAINT_CHANNEL,
    CandidatePlatformError,
    JointHostContract,
)

logger = logging.getLogger(__name__)


class JointConstrainMixin(JointHostContract):
    """Declared-constraint scoring for :class:`JointArchHwProblem`."""

    def onchip_fraction(self, configuration: Dict) -> float:
        """The share of this candidate's parameters that would sit on chip cores.

        Measured through the deployment's OWN estimator, under the CANDIDATE's
        placement — the NeuralOps/ComputeOps boundary the encoder lands on.
        """
        placement = self.candidate_encoding_placement(configuration)
        model, _params = self._candidate_model(
            configuration.get("model_config") or {},
            configuration.get("platform_constraints") or {},
            placement,
        )
        return estimate_onchip_fraction(
            model,
            tuple(self.input_shape),
            int(self.num_classes),
            encoding_placement=placement,
            metric="params",
        ).fraction

    def onchip_constraint(self, configuration: Dict) -> Optional[ConstraintReport]:
        """The on-chip floor report for this candidate, or None when satisfied."""
        if self.onchip_min_fraction <= 0.0:
            return None
        return onchip_floor_violation(
            fraction=self.onchip_fraction(configuration),
            floor=float(self.onchip_min_fraction),
        )

    def constraint_violation(self, configuration: Dict) -> float:
        try:
            resolved = self._resolved_configuration(configuration)
        except CandidatePlatformError:
            return 1.0
        declared = self._declared_violation(resolved, configuration)
        if declared is not None:
            return declared
        if not self.validate_detailed(resolved, channel=CONSTRAINT_CHANNEL).is_valid:
            return 1.0
        report = self.onchip_constraint(resolved)
        if report is None:
            return 0.0
        self._constraint_census[report.constraint] = (
            self._constraint_census.get(report.constraint, 0) + 1
        )
        logger.warning(
            "[JointArchHwProblem] candidate violates %s: %s",
            report.constraint, report.detail,
        )
        return report.violation

    def _declared_violation(
        self, resolved: Dict, configuration: Dict,
    ) -> Optional[float]:
        """The caller's own constraint reading, or None when it does not object.

        A cheap predicate over the DECLARATION: it builds nothing, so it costs
        the run no evaluation, and its own breakage is the candidate's problem
        rather than the search's.
        """
        if self.constraint_fn is None:
            return None
        try:
            cv = float(self.constraint_fn(
                resolved["model_config"],
                resolved["platform_constraints"],
                self.input_shape,
            ))
        except Exception as exc:
            logger.warning(
                "[JointArchHwProblem] constraint_fn failed (%s: %s) for candidate "
                "%.500s; recording constraint violation 1e6",
                type(exc).__name__, exc, configuration, exc_info=True,
            )
            return 1e6
        return cv if cv > 0 else None

    def constraint_census(self) -> Dict[str, int]:
        """How many candidates each declared constraint has rejected so far."""
        return dict(self._constraint_census)
