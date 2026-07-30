"""Baseline-agreement SSOT: a measured number that contradicts its record is a defect, not a result.

The defect this exists for: a model that normalizes its input INTERNALLY was fed
through an additional torchvision ``Normalize``. The harness reported 60.83%
against a recorded 92.30% and nothing raised -- the number was plausible,
self-consistent and wrong, and it was caught only because the reader happened to
know the expected value. That is the same failure family as the NaN-batch
corruption ``data_handling.batch_integrity`` guards: a rare fault that does not
crash, it emits a number.

The contract enforces the comparison and nothing else. It knows no workload, no
model and no dataset: the expectation is supplied by the caller or read from a
recorded artifact, and :class:`MetricTolerance` is the ONE place that decides
what "matches" means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

BASELINE_SIGMAS = 4.0
"""Sampling standard errors of the recorded proportion a measurement may sit from
it before the gap stops being explicable as noise."""

BASELINE_FLOOR = 0.01
"""Irreducible slack: a record and a re-measurement are different sample sets
(split, ordering, decode), so they are not expected to agree exactly even at
infinite sample count."""

LIKELY_CAUSES: tuple[str, ...] = (
    "the evaluation preprocessing is not the one the baseline was recorded "
    "under -- a model that normalizes its input internally, fed already "
    "normalized data, normalizes twice and still returns a plausible number",
    "the weights measured are not the ones the baseline names (state-dict keys "
    "silently skipped, or left at their initialization)",
    "the metric was measured on other data than the record describes (another "
    "split, subsample, dataset or label ordering)",
)


class BaselineMismatchError(AssertionError):
    """A measurement that contradicts the baseline it was recorded against.

    Deliberately not a ``ValueError``/``TypeError`` subclass, which callers
    routinely catch: a number that disagrees with its own record must end the
    measurement, not be degraded into a warning.
    """


@dataclass(frozen=True)
class MetricTolerance:
    """THE definition of "matches" for a measured number against a recorded one.

    Symmetric in its two arguments (the relative part scales with the larger
    magnitude), so a verdict never depends on argument order.
    """

    abs_tol: float = 0.0
    rel_tol: float = 0.0

    def __post_init__(self) -> None:
        for name in ("abs_tol", "rel_tol"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(
                    f"a tolerance must be finite and non-negative, got "
                    f"{name}={value!r}"
                )

    def bound(self, expected: float, measured: float) -> float:
        """The largest ``|measured - expected|`` this policy still calls a match."""
        return self.abs_tol + self.rel_tol * max(abs(float(expected)), abs(float(measured)))

    def matches(self, expected: float, measured: float) -> bool:
        """Whether the two values agree under this policy."""
        if expected == measured:
            return True
        return abs(float(measured) - float(expected)) <= self.bound(expected, measured)

    @classmethod
    def for_sampled_proportion(
        cls,
        expected: float,
        n_samples: int,
        *,
        sigmas: float = BASELINE_SIGMAS,
        floor: float = BASELINE_FLOOR,
    ) -> "MetricTolerance":
        """The tolerance a proportion measured on ``n_samples`` draws deserves.

        A tolerance quoted without the size of the sample it judges is a guess:
        the same gap is noise over 128 examples and a defect over 50,000.
        """
        n = int(n_samples)
        if n <= 0:
            raise ValueError(
                "a proportion measured on no samples is not a measurement, so "
                "no tolerance can judge it"
            )
        p = float(expected)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"a proportion baseline must lie in [0, 1], got {p!r}")
        standard_error = math.sqrt(p * (1.0 - p) / n)
        return cls(abs_tol=float(floor) + float(sigmas) * standard_error, rel_tol=0.0)


def _cause_list(causes: Sequence[str]) -> str:
    ordered = tuple(causes) + LIKELY_CAUSES
    return "; ".join(f"({i}) {cause}" for i, cause in enumerate(ordered, 1))


def assert_matches_recorded_baseline(
    measured: float,
    *,
    expected: float,
    tolerance: MetricTolerance,
    observable: str,
    recorded_as: str,
    causes: Sequence[str] = (),
) -> float:
    """Raise unless ``measured`` agrees with the recorded ``expected``; return ``measured``.

    Verification logic: it never degrades to a warning. ``causes`` are the
    call site's context-specific suspects, named before the generic ones.
    """
    measured_value = float(measured)
    expected_value = float(expected)
    if not math.isfinite(measured_value):
        raise BaselineMismatchError(
            f"{observable} came out {measured_value}, which is not a measurement "
            f"of anything -- the recorded baseline is {expected_value:.6f} "
            f"({recorded_as}). Most likely causes: {_cause_list(causes)}."
        )
    if tolerance.matches(expected_value, measured_value):
        return measured_value
    raise BaselineMismatchError(
        f"{observable} contradicts its recorded baseline: measured "
        f"{measured_value:.6f}, recorded {expected_value:.6f} ({recorded_as}) -- "
        f"gap {measured_value - expected_value:+.6f}, beyond the tolerance of "
        f"{tolerance.bound(expected_value, measured_value):.6f}. A number that "
        f"disagrees with its own record is a defect, not a result. Most likely "
        f"causes: {_cause_list(causes)}."
    )
