"""Semantic-axis equivalence-SCREEN instrument (Wave-9 A2).

The MEASURED resolution of whether a SEMANTIC knob (pruning, regime, quantization)
COLLAPSES or stays ENUMERATED for the coverage denominator. Distinct from
``cross_sim_parity`` (B1): a faithfulness axis is bit-exact and collapses on a
PARITY artifact, but a semantic knob CHANGES the trained result — so this screen
measures whether two cell-populations are EQUIVALENT (same validity tier AND
deployed accuracy within a tolerance band), not whether two simulators of one
contract agree.

Three outcome states per (axis, paired cell-group), keyed by the OTHER coordinates:

* ``EQUIVALENT`` — ``|Δacc| <= tol_pp`` AND the two cells share a validity tier;
  the semantic value does not move the deployed result → COLLAPSIBLE, with the
  MEASURED ``delta_pp`` recorded.
* ``INTERACTING`` — the band is exceeded OR the tier flips → STAYS ENUMERATED, with
  the MEASURED ``delta_pp`` (an upgrade from ``ASSERTED_UNSCREENED`` — the value is
  measured, not asserted).
* ``INSUFFICIENT_DATA`` — fewer than two distinct axis values are present in the
  group (no counterpart to pair against) → honestly unscreened.

The semantic axes are DERIVED from the ledger's own ``ScreeningStatus``
(``ASSERTED_UNSCREENED``) so this instrument never invents which knobs are
semantic. Rows are mapped to cells via ``coverage_ledger.row_to_cells`` and tiers
via ``classify_validity_tier`` — the cell/tier math is CONSUMED, not duplicated.

``write_semantic_screen`` emits a JSON-able artifact with NO wall-clock timestamp
(deterministic/diffable). ``assert_semantic_screen_sound`` is the honesty gate the
coverage screen calls before it may trust the artifact for a collapse: a collapse
claim REQUIRES an EQUIVALENT outcome with a recorded ``delta_pp``, a minimum cell
count, and no MEASURED interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.coverage_ledger import (
    AXES,
    ScreeningStatus,
    classify_validity_tier,
    row_to_cells,
)

__all__ = [
    "SemanticAxisState",
    "SemanticPairOutcome",
    "SemanticScreenError",
    "SEMANTIC_AXES",
    "screen_semantic_axis",
    "write_semantic_screen",
    "assert_semantic_screen_sound",
    "screen_live_regime",
    "screen_live_pruning",
]


class SemanticScreenError(AssertionError):
    """The semantic screen artifact is malformed or dishonestly claims collapse."""


class SemanticAxisState(Enum):
    """The 3-state outcome of one (axis, paired cell-group) screen."""

    EQUIVALENT = "equivalent"
    INTERACTING = "interacting"
    INSUFFICIENT_DATA = "insufficient_data"


_VALID_STATE_VALUES = frozenset(s.value for s in SemanticAxisState)

# The SEMANTIC axes — DERIVED from the ledger's own classification so this
# instrument never decides which knobs are semantic. A semantic knob CHANGES the
# trained result (it cannot collapse on a fidelity/parity artifact), so the ledger
# tags it ``ASSERTED_UNSCREENED`` until a measured equivalence screen earns better.
SEMANTIC_AXES: Tuple[str, ...] = tuple(
    axis.name
    for axis in AXES
    if axis.screening_status is ScreeningStatus.ASSERTED_UNSCREENED
)


@dataclass(frozen=True)
class SemanticPairOutcome:
    """One screened (axis, paired cell-group) result.

    ``group_key`` pins the OTHER coordinates the pair shares; ``value_a``/``value_b``
    are the two semantic-axis values compared. ``delta_pp`` is the MEASURED
    ``|Δacc|`` in percentage points (mandatory for EQUIVALENT and INTERACTING; ``None``
    for INSUFFICIENT_DATA — nothing was paired). ``reason`` carries the INTERACTING
    cause (band exceeded / tier flip, with the measured number) or the
    INSUFFICIENT_DATA cause; ``None`` for a plain EQUIVALENT.
    """

    axis: str
    group_key: str
    value_a: Optional[str]
    value_b: Optional[str]
    state: SemanticAxisState
    delta_pp: Optional[float]
    tol_pp: float
    tier_a: Optional[str]
    tier_b: Optional[str]
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "group_key": self.group_key,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "state": self.state.value,
            "delta_pp": self.delta_pp,
            "tol_pp": self.tol_pp,
            "tier_a": self.tier_a,
            "tier_b": self.tier_b,
            "reason": self.reason,
        }


# Per-sync deployed-accuracy field for a dual-schedule arch_dataset row; the single
# value falls back to the per-row keys, in priority order.
_SYNC_DEPLOYED_KEY = {
    "cascaded": "cascaded_deployed_mean",
    "synchronized": "synchronized_deployed_mean",
}
_SINGLE_DEPLOYED_KEYS = ("deployed_acc", "deployed_acc_mean")


def _deployed_acc_for_cell(row: Mapping[str, Any], sync: Optional[str]) -> Optional[float]:
    """The deployed accuracy a row reports for one of its cells (its ``sync``).

    A dual-schedule row carries a per-sync mean; a single-schedule row carries one
    of ``_SINGLE_DEPLOYED_KEYS``. Returns ``None`` when no deployed number is present
    (the cell cannot be screened on accuracy).
    """
    if sync in _SYNC_DEPLOYED_KEY:
        per_sync = row.get(_SYNC_DEPLOYED_KEY[sync])
        if per_sync is not None:
            return float(per_sync)
    for key in _SINGLE_DEPLOYED_KEYS:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


@dataclass(frozen=True)
class _Sample:
    """One (cell, axis-value, tier, deployed-accuracy) point feeding a pair group."""

    group_key: str
    axis_value: str
    tier: str
    deployed_acc: float


def _group_key_without_axis(cell, axis: str) -> str:
    """The cell's canonical key with the screened axis dropped — the pairing key.

    Two cells differing ONLY in the screened semantic axis share this key, so they
    land in the same pair group. Reuses the cell's own ``cell_key`` segments (the
    SSOT key) and strips the ``<axis>=<value>`` segment.
    """
    field_name = "S" if axis == "S" else axis
    parts = []
    for segment in cell.cell_key.split("|"):
        if segment.startswith(f"{field_name}="):
            continue
        parts.append(segment)
    return "|".join(parts)


def _samples_for_axis(rows: Sequence[Mapping[str, Any]], axis: str) -> List[_Sample]:
    """Map science-valid rows to ``_Sample`` points for the screened ``axis``.

    Each row expands to its cell(s) (``row_to_cells`` — a dual-schedule row → both
    sync cells); a cell with no validity tier or no deployed accuracy is skipped. The
    axis value is read from the cell's own coordinate so it matches the ledger's
    normalization (defaults included).
    """
    field_name = "s" if axis == "S" else axis
    samples: List[_Sample] = []
    for row in rows:
        tier = classify_validity_tier(row.get("deployment_validity"))
        if tier is None:
            continue
        for cell in row_to_cells(row):
            acc = _deployed_acc_for_cell(row, cell.sync)
            if acc is None:
                continue
            samples.append(
                _Sample(
                    group_key=_group_key_without_axis(cell, axis),
                    axis_value=str(getattr(cell, field_name)),
                    tier=str(row.get("deployment_validity")),
                    deployed_acc=acc,
                )
            )
    return samples


def _screen_group(
    axis: str, group_key: str, samples: Sequence[_Sample], *, tol_pp: float
) -> SemanticPairOutcome:
    """Screen one pairing group into a 3-state outcome.

    A group with fewer than two distinct axis values has no counterpart to pair →
    INSUFFICIENT_DATA. Otherwise the two extreme axis-value populations are compared
    on the WORST observed ``|Δacc|`` and tier agreement: EQUIVALENT iff the gap is
    within ``tol_pp`` AND both cells share a canonical validity tier, else INTERACTING
    with the measured gap and the cause.
    """
    distinct = sorted({s.axis_value for s in samples})
    if len(distinct) < 2:
        only = distinct[0] if distinct else "<none>"
        return SemanticPairOutcome(
            axis=axis,
            group_key=group_key,
            value_a=distinct[0] if distinct else None,
            value_b=None,
            state=SemanticAxisState.INSUFFICIENT_DATA,
            delta_pp=None,
            tol_pp=float(tol_pp),
            tier_a=None,
            tier_b=None,
            reason=(
                f"only {len(distinct)} distinct {axis!r} value(s) ({only}) in this "
                f"group — no counterpart to pair against; UNSCREENED, not collapsed"
            ),
        )

    value_a, value_b = distinct[0], distinct[-1]
    pop_a = [s for s in samples if s.axis_value == value_a]
    pop_b = [s for s in samples if s.axis_value == value_b]
    # The screen is conservative: take the WORST (largest) accuracy gap and report a
    # tier flip if ANY representative tier differs across the two populations.
    worst = max(
        (abs(a.deployed_acc - b.deployed_acc), a, b) for a in pop_a for b in pop_b
    )
    delta_pp = worst[0] * 100.0
    tier_a, tier_b = worst[1].tier, worst[2].tier
    canon_a = classify_validity_tier(tier_a)
    canon_b = classify_validity_tier(tier_b)
    tiers_agree = canon_a is not None and canon_a == canon_b

    band_ok = delta_pp <= float(tol_pp) + 1e-12
    if band_ok and tiers_agree:
        return SemanticPairOutcome(
            axis=axis,
            group_key=group_key,
            value_a=value_a,
            value_b=value_b,
            state=SemanticAxisState.EQUIVALENT,
            delta_pp=delta_pp,
            tol_pp=float(tol_pp),
            tier_a=tier_a,
            tier_b=tier_b,
            reason=None,
        )

    causes = []
    if not band_ok:
        causes.append(
            f"|Δacc|={delta_pp:.1f}pp exceeds the {float(tol_pp):.1f}pp band"
        )
    if not tiers_agree:
        causes.append(f"tier flip {canon_a} != {canon_b} ({tier_a} vs {tier_b})")
    return SemanticPairOutcome(
        axis=axis,
        group_key=group_key,
        value_a=value_a,
        value_b=value_b,
        state=SemanticAxisState.INTERACTING,
        delta_pp=delta_pp,
        tol_pp=float(tol_pp),
        tier_a=tier_a,
        tier_b=tier_b,
        reason="; ".join(causes),
    )


def screen_semantic_axis(
    axis: str, rows: Sequence[Mapping[str, Any]], *, tol_pp: float
) -> List[SemanticPairOutcome]:
    """Pair ledger rows ACROSS a semantic ``axis`` by the OTHER coordinates → outcomes.

    Each science-valid row maps to its cell(s); cells are grouped by every coordinate
    EXCEPT ``axis`` and each group is screened into a 3-state outcome. A group with no
    counterpart on the axis is reported INSUFFICIENT_DATA (never silently dropped). With
    no rows at all, one INSUFFICIENT_DATA outcome is returned so the artifact is honest.

    ``axis`` MUST be a SEMANTIC axis (``ASSERTED_UNSCREENED`` in the ledger) — a
    faithfulness axis (backend / mapping_strategy) collapses on ``cross_sim_parity``,
    not here, so passing one RAISES.
    """
    if axis not in SEMANTIC_AXES:
        raise SemanticScreenError(
            f"axis {axis!r} is not a SEMANTIC axis (the screenable knobs are "
            f"{list(SEMANTIC_AXES)}) — faithfulness axes collapse on cross_sim_parity, "
            f"not on a semantic equivalence screen"
        )

    samples = _samples_for_axis(rows, axis)
    if not samples:
        return [
            SemanticPairOutcome(
                axis=axis,
                group_key="<no-cells>",
                value_a=None,
                value_b=None,
                state=SemanticAxisState.INSUFFICIENT_DATA,
                delta_pp=None,
                tol_pp=float(tol_pp),
                tier_a=None,
                tier_b=None,
                reason="no science-valid cells with a deployed accuracy to screen",
            )
        ]

    grouped: Dict[str, List[_Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.group_key, []).append(sample)
    return [
        _screen_group(axis, key, grouped[key], tol_pp=tol_pp)
        for key in sorted(grouped)
    ]


def write_semantic_screen(
    axis: str,
    outcomes: Sequence[SemanticPairOutcome],
    *,
    tol_pp: float,
    methodology: str,
    justifies_collapse: bool = False,
    min_equivalent_cells: int = 1,
) -> Dict[str, Any]:
    """Build the JSON-able semantic-screen artifact dict.

    Deterministic / diffable: NO wall-clock timestamp. ``justifies_collapse`` records
    whether this artifact CLAIMS it supports collapsing ``axis`` (the soundness gate
    enforces that such a claim cannot coexist with a MEASURED interaction, missing Δ,
    or too few EQUIVALENT cells). ``min_equivalent_cells`` is the minimum EQUIVALENT
    cell count a collapse claim must clear.
    """
    counts: Dict[str, int] = {s.value: 0 for s in SemanticAxisState}
    for outcome in outcomes:
        counts[outcome.state.value] += 1
    return {
        "schema": "semantic_axis_screen/v1",
        "axis": axis,
        "tol_pp": float(tol_pp),
        "methodology": methodology,
        "justifies_collapse": bool(justifies_collapse),
        "min_equivalent_cells": int(min_equivalent_cells),
        "state_counts": counts,
        "outcomes": [o.to_dict() for o in outcomes],
    }


def assert_semantic_screen_sound(artifact: Mapping[str, Any]) -> None:
    """The honesty gate the coverage screen calls before trusting the artifact.

    RAISES :class:`SemanticScreenError` if:

    * any outcome state is malformed (not one of the 3 known states);
    * any EQUIVALENT or INTERACTING lacks a recorded ``delta_pp`` (an equivalence /
      interaction claim without a measured number is not honest);
    * the artifact claims ``justifies_collapse`` while it contains ANY MEASURED
      INTERACTING outcome (a semantic collapse cannot rest on a measured interaction);
    * the artifact claims ``justifies_collapse`` with fewer than ``min_equivalent_cells``
      EQUIVALENT outcomes (a collapse needs enough measured-equivalent evidence).
    """
    outcomes = artifact.get("outcomes")
    if not isinstance(outcomes, list):
        raise SemanticScreenError(
            "semantic screen malformed: 'outcomes' must be a list"
        )

    equivalent_count = 0
    interacting_count = 0
    for entry in outcomes:
        state = entry.get("state")
        group = entry.get("group_key")
        if state not in _VALID_STATE_VALUES:
            raise SemanticScreenError(
                f"semantic screen malformed: outcome state {state!r} is not one of "
                f"{sorted(_VALID_STATE_VALUES)} (group {group!r})"
            )
        delta_pp = entry.get("delta_pp")
        if state == SemanticAxisState.EQUIVALENT.value:
            equivalent_count += 1
            if delta_pp is None:
                raise SemanticScreenError(
                    f"semantic screen unsound: EQUIVALENT outcome for group {group!r} "
                    f"has no recorded delta_pp — an equivalence claim without a "
                    f"measured number is not honest"
                )
        elif state == SemanticAxisState.INTERACTING.value:
            interacting_count += 1
            if delta_pp is None:
                raise SemanticScreenError(
                    f"semantic screen unsound: INTERACTING outcome for group {group!r} "
                    f"has no recorded delta_pp — a measured interaction must be "
                    f"quantified (the upgrade from ASSERTED_UNSCREENED IS the number)"
                )
        else:  # INSUFFICIENT_DATA
            if delta_pp is not None:
                raise SemanticScreenError(
                    f"semantic screen malformed: INSUFFICIENT_DATA outcome for group "
                    f"{group!r} carries a delta_pp {delta_pp!r} — nothing was paired, "
                    f"so there is no measured number to record"
                )

    if not artifact.get("justifies_collapse"):
        return

    if interacting_count:
        raise SemanticScreenError(
            f"semantic screen dishonest: artifact claims justifies_collapse=True for "
            f"axis {artifact.get('axis')!r} but contains {interacting_count} MEASURED "
            f"INTERACTING outcome(s) — a semantic collapse cannot rest on a measured "
            f"interaction (it stays ENUMERATED)"
        )
    min_cells = int(artifact.get("min_equivalent_cells", 1))
    if equivalent_count < min_cells:
        raise SemanticScreenError(
            f"semantic screen dishonest: artifact claims justifies_collapse=True for "
            f"axis {artifact.get('axis')!r} with only {equivalent_count} EQUIVALENT "
            f"outcome(s) — fewer than the min_equivalent_cells={min_cells} a collapse "
            f"needs (not enough measured-equivalent evidence)"
        )


def _screen_live_axis(
    axis: str, rows: Sequence[Mapping[str, Any]], *, tol_pp: float, methodology: str
) -> Dict[str, Any]:
    """Apply the screen to the LIVE ledger for one semantic axis (never claims collapse).

    Records the MEASURED state — INSUFFICIENT_DATA while the paired rows are draining,
    or the measured EQUIVALENT/INTERACTING per group. This is the INSTRUMENT step; it
    does NOT flip the axis in ``coverage_ledger`` (that is the later consume step once
    real paired data exists), so ``justifies_collapse`` is always ``False`` here.
    """
    outcomes = screen_semantic_axis(axis, rows, tol_pp=tol_pp)
    return write_semantic_screen(
        axis,
        outcomes,
        tol_pp=tol_pp,
        methodology=methodology,
        justifies_collapse=False,
    )


def screen_live_regime(
    rows: Sequence[Mapping[str, Any]], *, tol_pp: float = 1.0
) -> Dict[str, Any]:
    """Screen the LIVE ledger for the ``regime`` axis (from_scratch ↔ pretrained).

    The F3 dual-regime rows are draining: while too few paired cells exist the
    outcomes are honestly INSUFFICIENT_DATA; once both regimes are present per group
    the MEASURED Δ/tier outcome is recorded. Never claims collapse (the AXIS flip is
    the later consume step).
    """
    return _screen_live_axis(
        "regime",
        rows,
        tol_pp=tol_pp,
        methodology=(
            "pair from_scratch↔pretrained cells by the OTHER coordinates; EQUIVALENT "
            "iff |Δdeployed-acc| within tol_pp AND same validity tier. The F3 "
            "dual-regime rows are draining — INSUFFICIENT_DATA is reported honestly "
            "until both regimes are present per cell-group."
        ),
    )


def screen_live_pruning(
    rows: Sequence[Mapping[str, Any]], *, tol_pp: float = 1.0
) -> Dict[str, Any]:
    """Screen the LIVE ledger for the ``pruning`` axis (dense ↔ pruned).

    No pruned rows exist yet, so every group is honestly INSUFFICIENT_DATA. The
    instrument ships now; once a P3 pruning campaign emits pruned rows the same screen
    measures the Δ/tier outcome. Never claims collapse.
    """
    return _screen_live_axis(
        "pruning",
        rows,
        tol_pp=tol_pp,
        methodology=(
            "pair dense↔pruned cells by the OTHER coordinates; EQUIVALENT iff "
            "|Δdeployed-acc| within tol_pp AND same validity tier. No pruned rows "
            "exist yet — INSUFFICIENT_DATA is reported honestly until a P3 pruning "
            "campaign emits the counterpart cells."
        ),
    )
