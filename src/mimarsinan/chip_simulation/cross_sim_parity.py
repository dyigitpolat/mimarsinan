"""Cross-simulator parity SCREENING instrument (Wave-1 B1).

A denominator-shrinking screening ARTIFACT for the ``backend`` hypervolume axis.
A later coverage step may flip ``backend`` from ``ASSERTED_UNSCREENED`` to
``SCREENED_COLLAPSED`` (fidelity-only, like ``encoding_placement``) — but ONLY
because this instrument RECORDS measured equivalence per (cell, backend-pair); it
never ASSERTS it.

Three outcome states per (cell, backend-pair):

* ``AGREE`` — the two backends ran the cell and their per-neuron values match
  within ``tolerance``; the MEASURED ``max_abs_diff`` is recorded.
* ``DISAGREE`` — they ran the cell but the value gap exceeds ``tolerance``; the
  quantified gap is recorded (and a ``reason`` if the gap is understood/expected).
* ``INAPPLICABLE`` — a backend cannot run the cell's spiking mode (DERIVED from the
  ``_BACKEND_CAPS`` capability registry via ``SpikingModePolicy.supports_backend``).
  The backend is NEVER executed-to-hang; the reason is recorded.

The nevresim Python sim + the HCM/SCM analytical reference are the FAST,
already-bit-exact pair (the torch↔sim fidelity lock proves it cell-by-cell). The
parity MATH is REUSED from ``pipelining.core.nf_scm_parity`` (the order-insensitive
per-perceptron sorted-multiset comparison), never re-implemented here.

SANA-FE / Lava are screened by CAPABILITY only: their applicability is derived from
the registry and recorded as INAPPLICABLE-with-reason when a mode is unsupported —
they are not run by this instrument.

``write_cross_sim_screen`` emits a JSON-able artifact with NO wall-clock timestamp
(deterministic/diffable). ``assert_cross_sim_screen_sound`` is the honesty gate the
coverage screen calls before it may trust the artifact for a collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.pipelining.core.nf_scm_parity import compare_normalized_records

__all__ = [
    "CrossSimState",
    "CrossSimOutcome",
    "CrossSimParityError",
    "derive_applicability",
    "measured_max_abs_diff",
    "screen_cell_pair",
    "write_cross_sim_screen",
    "assert_cross_sim_screen_sound",
]


class CrossSimParityError(AssertionError):
    """The cross-sim screening artifact is malformed or dishonestly claims collapse."""


class CrossSimState(Enum):
    """The 3-state outcome of one (cell, backend-pair) screen."""

    AGREE = "agree"
    DISAGREE = "disagree"
    INAPPLICABLE = "inapplicable"


_VALID_STATE_VALUES = frozenset(s.value for s in CrossSimState)


@dataclass(frozen=True)
class CrossSimOutcome:
    """One screened (cell, backend-pair) result.

    ``max_abs_diff`` is the MEASURED per-neuron value gap (mandatory for AGREE and
    DISAGREE; ``None`` for INAPPLICABLE — nothing was run). ``reason`` carries the
    INAPPLICABLE cause or a DISAGREE explanation (``None`` for a plain AGREE and for
    an un-reasoned DISAGREE the soundness gate must catch under a collapse claim).
    """

    cell: str
    backend_pair: Tuple[str, str]
    state: CrossSimState
    max_abs_diff: Optional[float]
    tolerance: float
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell": self.cell,
            "backend_pair": [self.backend_pair[0], self.backend_pair[1]],
            "state": self.state.value,
            "max_abs_diff": self.max_abs_diff,
            "tolerance": self.tolerance,
            "reason": self.reason,
        }


def derive_applicability(
    backend: str, spiking_mode: str
) -> Tuple[bool, Optional[str]]:
    """Whether ``backend`` can run ``spiking_mode`` — DERIVED from the capability
    registry, never by executing the backend.

    Returns ``(applicable, reason)``: a supported pair is ``(True, None)``; an
    unsupported pair is ``(False, "<backend> cannot run spiking_mode=<mode> ...")``.
    """
    policy = policy_for_spiking_mode(spiking_mode)
    if policy.supports_backend(backend):
        return True, None
    reason = (
        f"backend {backend!r} cannot run spiking_mode={spiking_mode!r} "
        f"(unsupported in the capability registry) — screened INAPPLICABLE, "
        f"not executed"
    )
    return False, reason


def measured_max_abs_diff(
    nf_record: Dict[int, np.ndarray],
    scm_record: Dict[int, np.ndarray],
) -> float:
    """The max per-neuron |Δ| between two per-perceptron records.

    REUSES the order-insensitive sorted-multiset comparison from
    ``nf_scm_parity.compare_normalized_records`` (atol=0 reports the true worst
    gap; a permuted-but-equal row reads 0). The parity math is wrapped, not
    duplicated.
    """
    _, _, worst = compare_normalized_records(nf_record, scm_record, atol=0.0)
    if worst is None:
        return 0.0
    return float(worst[0])


def screen_cell_pair(
    *,
    cell: str,
    backend_a: str,
    backend_b: str,
    spiking_mode: str,
    nf_record: Optional[Dict[int, np.ndarray]],
    scm_record: Optional[Dict[int, np.ndarray]],
    tolerance: float,
    disagree_reason: Optional[str] = None,
) -> CrossSimOutcome:
    """Screen one (cell, backend-pair) into a 3-state outcome.

    Applicability is checked FIRST from the capability registry: if EITHER backend
    cannot run the mode, the outcome is INAPPLICABLE (the records are not consulted
    and the backend is never run). Otherwise the supplied per-neuron records are
    compared: AGREE if the measured ``max_abs_diff <= tolerance``, else DISAGREE
    (carrying ``disagree_reason`` if the gap is understood).
    """
    applicable_a, reason_a = derive_applicability(backend_a, spiking_mode)
    applicable_b, reason_b = derive_applicability(backend_b, spiking_mode)
    if not applicable_a or not applicable_b:
        reason = reason_a if not applicable_a else reason_b
        return CrossSimOutcome(
            cell=cell,
            backend_pair=(backend_a, backend_b),
            state=CrossSimState.INAPPLICABLE,
            max_abs_diff=None,
            tolerance=tolerance,
            reason=reason,
        )

    if nf_record is None or scm_record is None:
        raise CrossSimParityError(
            f"cell {cell!r} pair ({backend_a!r}, {backend_b!r}) is applicable for "
            f"spiking_mode={spiking_mode!r} but no records were provided to compare "
            f"— an applicable pair must be RUN and measured, not skipped"
        )

    diff = measured_max_abs_diff(nf_record, scm_record)
    if diff <= float(tolerance):
        return CrossSimOutcome(
            cell=cell,
            backend_pair=(backend_a, backend_b),
            state=CrossSimState.AGREE,
            max_abs_diff=diff,
            tolerance=tolerance,
            reason=None,
        )
    return CrossSimOutcome(
        cell=cell,
        backend_pair=(backend_a, backend_b),
        state=CrossSimState.DISAGREE,
        max_abs_diff=diff,
        tolerance=tolerance,
        reason=disagree_reason,
    )


def _backend_pairs(outcomes: Sequence[CrossSimOutcome]) -> list[list[str]]:
    """The distinct backend pairs the screen covered, in first-seen order."""
    seen: list[list[str]] = []
    for o in outcomes:
        pair = [o.backend_pair[0], o.backend_pair[1]]
        if pair not in seen:
            seen.append(pair)
    return seen


def write_cross_sim_screen(
    outcomes: Sequence[CrossSimOutcome],
    *,
    tolerance: float,
    methodology: str,
    justifies_collapse: bool = False,
) -> Dict[str, Any]:
    """Build the JSON-able screening artifact dict.

    Deterministic / diffable: NO wall-clock timestamp is embedded. ``methodology``
    notes WHY the nevresim≡HCM pair is the fast already-bit-exact reference and that
    SANA-FE/Lava applicability is capability-derived. ``justifies_collapse`` records
    whether this artifact CLAIMS it supports collapsing the ``backend`` axis (the
    soundness gate enforces that such a claim cannot coexist with an un-reasoned
    DISAGREE).
    """
    return {
        "schema": "cross_sim_parity/v1",
        "tolerance": float(tolerance),
        "methodology": methodology,
        "justifies_collapse": bool(justifies_collapse),
        "backend_pairs": _backend_pairs(outcomes),
        "outcomes": [o.to_dict() for o in outcomes],
    }


def assert_cross_sim_screen_sound(artifact: Dict[str, Any]) -> None:
    """The honesty gate the coverage screen calls before trusting the artifact.

    RAISES :class:`CrossSimParityError` if:

    * any outcome state is malformed (not one of the 3 known states);
    * any AGREE or DISAGREE lacks a recorded ``max_abs_diff`` (an equivalence /
      gap claim without a measured number is not honest);
    * any INAPPLICABLE carries a ``max_abs_diff`` (nothing was run, so there is no
      number to record) or lacks a reason;
    * the artifact claims it ``justifies_collapse`` while containing an un-reasoned
      DISAGREE (a collapse cannot rest on an unexplained divergence).
    """
    outcomes = artifact.get("outcomes")
    if not isinstance(outcomes, list):
        raise CrossSimParityError(
            "cross-sim screen malformed: 'outcomes' must be a list"
        )

    unreasoned_disagrees: list[str] = []
    for entry in outcomes:
        state = entry.get("state")
        if state not in _VALID_STATE_VALUES:
            raise CrossSimParityError(
                f"cross-sim screen malformed: outcome state {state!r} is not one of "
                f"{sorted(_VALID_STATE_VALUES)} (cell {entry.get('cell')!r})"
            )
        max_abs_diff = entry.get("max_abs_diff")
        cell = entry.get("cell")
        if state == CrossSimState.AGREE.value:
            if max_abs_diff is None:
                raise CrossSimParityError(
                    f"cross-sim screen unsound: AGREE outcome for cell {cell!r} "
                    f"pair {entry.get('backend_pair')!r} has no recorded "
                    f"max_abs_diff — an equivalence claim without a measured number "
                    f"is not honest"
                )
        elif state == CrossSimState.DISAGREE.value:
            if max_abs_diff is None:
                raise CrossSimParityError(
                    f"cross-sim screen unsound: DISAGREE outcome for cell {cell!r} "
                    f"pair {entry.get('backend_pair')!r} has no recorded "
                    f"max_abs_diff — a divergence must be quantified"
                )
            if not (entry.get("reason") and str(entry["reason"]).strip()):
                unreasoned_disagrees.append(str(cell))
        else:  # INAPPLICABLE
            if max_abs_diff is not None:
                raise CrossSimParityError(
                    f"cross-sim screen malformed: INAPPLICABLE outcome for cell "
                    f"{cell!r} carries a max_abs_diff {max_abs_diff!r} — nothing was "
                    f"run, so there is no measured number to record"
                )
            if not (entry.get("reason") and str(entry["reason"]).strip()):
                raise CrossSimParityError(
                    f"cross-sim screen unsound: INAPPLICABLE outcome for cell "
                    f"{cell!r} has no reason — applicability must name why the "
                    f"backend cannot run the mode"
                )

    if artifact.get("justifies_collapse") and unreasoned_disagrees:
        raise CrossSimParityError(
            f"cross-sim screen dishonest: artifact claims justifies_collapse=True "
            f"but contains {len(unreasoned_disagrees)} un-reasoned DISAGREE "
            f"outcome(s) {unreasoned_disagrees} — a collapse cannot rest on an "
            f"unexplained backend divergence"
        )
