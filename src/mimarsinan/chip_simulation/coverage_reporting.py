"""Coverage aggregation: GROUP BY hypervolume cell x validity tier into a CoverageReport."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.coverage_rows import (
    PLACEMENT_FIXABLE_DEFAULT_OWNER,
    PLACEMENT_FIXABLE_FIX_PATH,
    TIER_SEVERITY,
    CoverageStatus,
    classify_validity_tier,
    flag_owner_of,
    flag_ts_of,
    is_placement_fixable_flag,
    mine_flagged_ops,
    parse_ledger_timestamp,
    row_to_cells,
)
from mimarsinan.chip_simulation.hypervolume_cells import HypervolumeCell, cell_covers


class AttributionFidelity(Enum):
    """How trustworthy a covered region's per-neuron attribution is.

    ``ATTRIBUTION`` — bit-exact per-neuron reassembly; ``VALUE_DOMAIN_ONLY`` — only the
    deployed-accuracy value domain is bit-exact, the attribution reassembly is not gated in production.
    """

    ATTRIBUTION = "attribution"
    VALUE_DOMAIN_ONLY = "value_domain_only"


KNOWN_CRACKED_REGIONS: Tuple[str, ...] = (
    (
        "GAP-1: coalescing+output-tiling per-neuron attribution at VGG scale — "
        "C3 fixed the fidelity-harness reassembler keying (joint (output_slice, ir_id) "
        "is bit-exact in tests/integration/_split_reassembly.py); the PRODUCTION "
        "NF↔SCM gate stays identity-mapping-only, so the fragment path is NOT exercised "
        "in deployment"
    ),
    "residual Tier-1 merge (fused-mapping reassembly)",
)


def _attribution_fidelity_map() -> Dict[str, AttributionFidelity]:
    """The per-region attribution-fidelity map, marking the KNOWN-CRACKED regions ``VALUE_DOMAIN_ONLY``."""
    return {region: AttributionFidelity.VALUE_DOMAIN_ONLY for region in KNOWN_CRACKED_REGIONS}


@dataclass(frozen=True)
class FlagMetadata:
    """Owner + aging metadata for one VALID_FLAGGED cell — the flag-aging schema.

    ``owner`` ``None`` ⇒ UNOWNED; ``age_days`` is measured vs the report's ``now_ts``;
    ``fix_path`` is set when the flag has a KNOWN fix (a placement-fixable offload flip).
    """

    cell_key: str
    owner: Optional[str]
    flag_ts: Optional[str]
    age_days: Optional[int]
    fix_path: Optional[str] = None

    @property
    def is_unowned(self) -> bool:
        return not (self.owner and str(self.owner).strip())


@dataclass(frozen=True)
class CoverageReport:
    """The measured coverage of a claimed sub-product over a ledger.

    ``cell_status`` is the GROUP BY result (each covered cell → its worst tier);
    ``coverage_fraction`` = covered / claimed. ``research_gap_frontier`` and
    ``placement_fixable_frontier`` union the respective ops over VALID_FLAGGED cells.
    """

    cell_status: Dict[str, CoverageStatus]
    tier_counts: Dict[CoverageStatus, int]
    research_gap_frontier: List[str]
    placement_fixable_frontier: List[str] = field(default_factory=list)
    claimed_cells: Tuple[HypervolumeCell, ...] = ()
    flag_metadata: Tuple[FlagMetadata, ...] = ()
    attribution_fidelity: Dict[str, AttributionFidelity] = field(default_factory=dict)
    _covered_keys: frozenset = field(default_factory=frozenset)

    @property
    def claimed_subproduct_size(self) -> int:
        """The claimed sub-product SIZE — the denominator printed next to the fraction."""
        return len(self.claimed_cells)

    @property
    def aged_unowned_flags(self) -> List[FlagMetadata]:
        """Flagged cells that have NO owner (regardless of age) — the aging worklist."""
        return [m for m in self.flag_metadata if m.is_unowned]

    def _matching_statuses(self, claimed: HypervolumeCell) -> List[CoverageStatus]:
        """The tiers of every COVERED cell that satisfies ``claimed`` (wildcard-aware)."""
        statuses: List[CoverageStatus] = []
        for key, status in self.cell_status.items():
            if cell_covers(claimed, HypervolumeCell.from_key(key)):
                statuses.append(status)
        return statuses

    @property
    def covered_cell_count(self) -> int:
        return len(self.cell_status)

    @property
    def claimed_cell_count(self) -> int:
        return len(self.claimed_cells)

    @property
    def covered_claimed_count(self) -> int:
        return sum(1 for c in self.claimed_cells if self._matching_statuses(c))

    @property
    def coverage_fraction(self) -> float:
        if not self.claimed_cells:
            return 0.0
        return self.covered_claimed_count / self.claimed_cell_count

    @property
    def untested_frontier(self) -> List[HypervolumeCell]:
        return [c for c in self.claimed_cells if not self._matching_statuses(c)]

    def status_for(self, cell: HypervolumeCell) -> CoverageStatus:
        matches = self._matching_statuses(cell)
        if not matches:
            return CoverageStatus.UNTESTED
        return max(matches, key=lambda s: TIER_SEVERITY[s])

    def to_dict(self) -> Dict[str, Any]:
        # No merged valid-tier headline: VALID and VALID_FLAGGED are always reported separately.
        return {
            "covered_cell_count": self.covered_cell_count,
            "claimed_cell_count": self.claimed_cell_count,
            "claimed_subproduct_size": self.claimed_subproduct_size,
            "covered_claimed_count": self.covered_claimed_count,
            "coverage_fraction": self.coverage_fraction,
            "tier_counts": {s.value: n for s, n in self.tier_counts.items()},
            "cell_status": {k: v.value for k, v in sorted(self.cell_status.items())},
            "untested_frontier": [c.cell_key for c in self.untested_frontier],
            "research_gap_frontier": list(self.research_gap_frontier),
            "placement_fixable_frontier": list(self.placement_fixable_frontier),
            "attribution_fidelity": {
                region: fid.value for region, fid in self.attribution_fidelity.items()
            },
            "flag_metadata": [
                {
                    "cell_key": m.cell_key,
                    "owner": m.owner,
                    "flag_ts": m.flag_ts,
                    "age_days": m.age_days,
                    "fix_path": m.fix_path,
                }
                for m in self.flag_metadata
            ],
        }


def coverage_report(
    rows: Iterable[Mapping[str, Any]],
    claimed_subproduct: Optional[Sequence[HypervolumeCell]] = None,
    now_ts: Optional[str] = None,
) -> CoverageReport:
    """GROUP BY the ledger by hypervolume cell-key + validity tier → coverage.

    A cell's status is the WORST tier of its rows; the frontiers union the flag ops over
    VALID_FLAGGED cells. When ``claimed_subproduct`` is given, coverage is measured against it.
    """
    materialized = list(rows)
    now = parse_ledger_timestamp(now_ts) or _dt.date.today()

    cell_status: Dict[str, CoverageStatus] = {}
    row_cells: List[Tuple[Mapping[str, Any], CoverageStatus, List[HypervolumeCell]]] = []
    for row in materialized:
        tier = classify_validity_tier(row.get("deployment_validity"))
        if tier is None:
            continue
        cells = row_to_cells(row)
        if not cells:
            continue
        row_cells.append((row, tier, cells))
        for cell in cells:
            key = cell.cell_key
            prior = cell_status.get(key)
            if prior is None or TIER_SEVERITY[tier] > TIER_SEVERITY[prior]:
                cell_status[key] = tier

    research_gaps: set = set()
    placement_fixable: set = set()
    flag_meta: Dict[str, FlagMetadata] = {}
    for row, tier, cells in row_cells:
        if tier is not CoverageStatus.VALID_FLAGGED:
            continue
        flagged_cells = [
            cell
            for cell in cells
            if cell_status[cell.cell_key] is CoverageStatus.VALID_FLAGGED
        ]
        if not flagged_cells:
            continue
        gaps, placement = mine_flagged_ops(row)
        research_gaps.update(gaps)
        placement_fixable.update(placement)
        owner = flag_owner_of(row)
        fix_path = None
        if is_placement_fixable_flag(row):
            fix_path = PLACEMENT_FIXABLE_FIX_PATH
            if owner is None:
                owner = PLACEMENT_FIXABLE_DEFAULT_OWNER
        flag_ts = flag_ts_of(row)
        flagged_on = parse_ledger_timestamp(flag_ts)
        age_days = (now - flagged_on).days if flagged_on is not None else None
        for cell in flagged_cells:
            flag_meta.setdefault(
                cell.cell_key,
                FlagMetadata(
                    cell_key=cell.cell_key,
                    owner=owner,
                    flag_ts=flag_ts,
                    age_days=age_days,
                    fix_path=fix_path,
                ),
            )

    tier_counts: Dict[CoverageStatus, int] = {
        CoverageStatus.VALID: 0,
        CoverageStatus.VALID_FLAGGED: 0,
        CoverageStatus.INVALID: 0,
    }
    for status in cell_status.values():
        tier_counts[status] += 1

    covered_keys = frozenset(cell_status)
    claimed = tuple(claimed_subproduct or ())

    return CoverageReport(
        cell_status=cell_status,
        tier_counts=tier_counts,
        research_gap_frontier=sorted(research_gaps),
        placement_fixable_frontier=sorted(placement_fixable),
        claimed_cells=claimed,
        flag_metadata=tuple(flag_meta[k] for k in sorted(flag_meta)),
        attribution_fidelity=_attribution_fidelity_map(),
        _covered_keys=covered_keys,
    )
