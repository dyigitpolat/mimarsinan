"""Cross-simulator parity screening instrument recording measured per-(cell, backend-pair) equivalence."""

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

    ``max_abs_diff`` is the MEASURED per-neuron gap (``None`` for INAPPLICABLE — nothing
    was run); ``reason`` carries the INAPPLICABLE cause or a DISAGREE explanation.
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
    """Whether ``backend`` can run ``spiking_mode``, derived from the capability registry (never executed).

    Returns ``(applicable, reason)`` — ``(True, None)`` when supported, else ``(False, <why>)``.
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
    """The max per-neuron |Δ| between two per-perceptron records, via ``nf_scm_parity`` (wrapped, not duplicated).

    Records must already share the deployed neuron reality (a pruned side must first be
    projected with ``mapping.pruning.DeployedNeuronSurvival.project``), else the shape check fails loud.
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

    Applicability is checked FIRST: if either backend cannot run the mode → INAPPLICABLE
    (records not consulted). Otherwise AGREE iff ``max_abs_diff <= tolerance``, else DISAGREE.
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
    """Build the JSON-able screening artifact dict (deterministic/diffable — no timestamp).

    ``justifies_collapse`` records whether the artifact claims it supports collapsing the
    ``backend`` axis (the soundness gate rejects that claim over an un-reasoned DISAGREE).
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

    RAISES on a malformed state, an AGREE/DISAGREE without a measured ``max_abs_diff``, an
    INAPPLICABLE that carries one or lacks a reason, or a collapse claim over an un-reasoned DISAGREE.
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
        else:
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
