"""Ledger-row interpretation: validity tiers, covered cells, timestamps, and flag-op mining."""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mimarsinan.chip_simulation.hypervolume_axes import AXIS_WILDCARD
from mimarsinan.chip_simulation.hypervolume_axis_encoder import (
    cell_coordinates_from_row,
    syncs_from_row,
)
from mimarsinan.chip_simulation.hypervolume_cells import HypervolumeCell
from mimarsinan.mapping.verification.onchip_fraction import (
    TIER_INVALID,
    TIER_VALID,
    TIER_VALID_FLAGGED,
)


class CoverageStatus(Enum):
    """The per-cell coverage status: VALID / VALID_FLAGGED / INVALID (worst tier wins) or UNTESTED (no row)."""

    VALID = TIER_VALID
    VALID_FLAGGED = TIER_VALID_FLAGGED
    INVALID = TIER_INVALID
    UNTESTED = "UNTESTED"


TIER_SEVERITY: Dict[CoverageStatus, int] = {
    CoverageStatus.INVALID: 3,
    CoverageStatus.VALID_FLAGGED: 2,
    CoverageStatus.VALID: 1,
}


def classify_validity_tier(raw: Optional[str]) -> Optional[CoverageStatus]:
    """Normalize a free-form ledger ``deployment_validity`` string to a canonical tier (``None`` for a non-science row).

    Order matters: ``VALID_FLAGGED`` and ``INVALID`` are checked before the ``VALID`` prefix.
    """
    if not raw:
        return None
    text = str(raw).upper()
    if "INVALID" in text:
        return CoverageStatus.INVALID
    if "FLAGGED" in text:
        return CoverageStatus.VALID_FLAGGED
    if text.startswith("VALID"):
        return CoverageStatus.VALID
    return None


def _cell_with_sync(row: Mapping[str, Any], vehicle: str, sync: str) -> HypervolumeCell:
    coords = cell_coordinates_from_row(row, sync=sync, axis_wildcard=AXIS_WILDCARD)
    data = coords.as_cell_kwargs()
    data["vehicle"] = str(vehicle)
    return HypervolumeCell(**data)


def row_to_cells(row: Mapping[str, Any]) -> List[HypervolumeCell]:
    """Map one science-valid ledger row to the hypervolume cell(s) it covers (a dual-schedule row → both sync cells).

    Returns ``[]`` when the row carries no validity tier or no model.
    """
    if classify_validity_tier(row.get("deployment_validity")) is None:
        return []
    vehicle = row.get("model") or row.get("model_type")
    if not vehicle:
        return []
    return [_cell_with_sync(row, vehicle, sync) for sync in syncs_from_row(row)]


def row_to_cell(row: Mapping[str, Any]) -> Optional[HypervolumeCell]:
    """The first hypervolume cell a row covers (``None`` for a non-science row)."""
    cells = row_to_cells(row)
    return cells[0] if cells else None


def parse_ledger_timestamp(value: Any) -> Optional[_dt.date]:
    """Parse an ISO ``YYYY-MM-DD`` string OR a Unix-epoch number (the ledger writes both) to a date."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return _dt.datetime.utcfromtimestamp(float(value)).date()
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    try:
        return _dt.datetime.utcfromtimestamp(float(text)).date()
    except (ValueError, OverflowError, OSError):
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return _dt.datetime.strptime(text[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    try:
        return _dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


_PLACEMENT_FLAG_MARKER = "PLACEMENT"

PLACEMENT_FIXABLE_DEFAULT_OWNER = "program:placement-offload"
PLACEMENT_FIXABLE_FIX_PATH = "set encoding_layer_placement=offload"


def is_placement_fixable_flag(row: Mapping[str, Any]) -> bool:
    """True iff a VALID_FLAGGED row's flag is PLACEMENT-FIXABLE (a placement_fixable_ops encoder with no research gap)."""
    gaps, placement = mine_flagged_ops(row)
    return bool(placement) and not gaps


def mine_flagged_ops(row: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """``(research_gap_ops, placement_fixable_ops)`` named by one VALID_FLAGGED row.

    Prefers the structured fields; falls back to deriving the category from the
    ``deployment_validity`` tier suffix for a live ledger row that predates them.
    """
    structured_gaps = [str(op) for op in (row.get("research_gap_ops") or ())]
    structured_placement = [str(op) for op in (row.get("placement_fixable_ops") or ())]
    if structured_gaps or structured_placement:
        return structured_gaps, structured_placement

    text = str(row.get("deployment_validity") or "").upper()
    if _PLACEMENT_FLAG_MARKER in text:
        return [], ["encoding_layer(placement)"]
    return ["unsupported_host_op"], []


def flag_owner_of(row: Mapping[str, Any]) -> Optional[str]:
    """The flag owner named by a row (``flag_owner`` / ``owner``), else ``None``."""
    owner = row.get("flag_owner") or row.get("owner")
    owner = str(owner).strip() if owner is not None else ""
    return owner or None


def flag_ts_of(row: Mapping[str, Any]) -> Optional[str]:
    """The flag timestamp named by a row (``flag_ts`` / ``ts``), else ``None``."""
    ts = row.get("flag_ts") or row.get("ts")
    return str(ts) if ts else None
