"""Per-cell regression-floor freezing + gating on the deployed-forward metric and wall-clock budget."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional


__all__ = [
    "CertificationCell",
    "RegressionFloor",
    "CertificationFloorBook",
    "CertificationStatus",
    "CertificationVerdict",
    "AbsoluteVerdict",
    "DEFAULT_ACCURACY_EPS",
    "DEFAULT_WALL_CLOCK_SLACK",
    "freeze_cell",
    "certify",
    "floor_is_stale",
    "load_floor_book",
    "save_floor_book",
    "FLOOR_BOOK_FORMAT_VERSION",
]


FLOOR_BOOK_FORMAT_VERSION = 1

DEFAULT_ACCURACY_EPS = 0.0
DEFAULT_WALL_CLOCK_SLACK = 0.0


@dataclass(frozen=True)
class CertificationCell:
    """The matrix coordinate the per-cell gate is keyed by: (firing × sync × backend).

    ``sync`` is the ``ttfs_cycle_schedule`` (or ``None`` for non-cycle families);
    ``variant`` optionally separates the floors of two configs sharing a recipe cell.
    """

    firing: str
    sync: Optional[str]
    backend: str
    variant: Optional[str] = None

    @property
    def cell_key(self) -> str:
        """Canonical string key — ``mode[/schedule]@backend[#variant]``."""
        mode = self.firing if self.sync is None else f"{self.firing}/{self.sync}"
        key = f"{mode}@{self.backend}"
        return key if self.variant is None else f"{key}#{self.variant}"

    @classmethod
    def from_key(cls, key: str) -> "CertificationCell":
        """Parse a canonical ``mode[/schedule]@backend[#variant]`` key into a cell."""
        if "@" not in key:
            raise ValueError(
                f"certification cell key {key!r} is missing the '@backend' suffix"
            )
        variant = None
        if "#" in key:
            key, variant = key.rsplit("#", 1)
            if not variant:
                raise ValueError(f"certification cell key has an empty '#variant'")
        mode_part, backend = key.rsplit("@", 1)
        if "/" in mode_part:
            firing, sync = mode_part.split("/", 1)
        else:
            firing, sync = mode_part, None
        if not firing or not backend:
            raise ValueError(f"certification cell key {key!r} is malformed")
        return cls(firing=firing, sync=sync, backend=backend, variant=variant)

    @classmethod
    def from_mode_policy(
        cls, mode_policy: Any, *, backend: str, variant: Optional[str] = None
    ) -> "CertificationCell":
        """Build a cell from a (firing × sync) ``SpikingModePolicy`` + a backend."""
        firing = str(getattr(mode_policy, "spiking_mode", "lif"))
        sync = getattr(mode_policy, "schedule", None)
        sync = None if sync is None else str(sync)
        return cls(firing=firing, sync=sync, backend=str(backend), variant=variant)


@dataclass(frozen=True)
class RegressionFloor:
    """The frozen deployed floor for one cell — the regression baseline the gate uses.

    ``deployed_accuracy`` / ``wall_clock_s`` are the frozen numbers; ``eps`` /
    ``wall_clock_slack`` / ``wall_clock_budget_s`` the per-cell tolerances. The
    optional F1 ``ac*`` targets default ``None`` ⇒ the floor round-trips byte-identically.
    """

    deployed_accuracy: float
    wall_clock_s: float
    eps: float = DEFAULT_ACCURACY_EPS
    wall_clock_slack: float = DEFAULT_WALL_CLOCK_SLACK
    wall_clock_budget_s: Optional[float] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    ac1_target: Optional[float] = None
    ac2_reference: Optional[float] = None
    ac5_budget_s: Optional[float] = None

    def accuracy_floor(self) -> float:
        """The lowest deployed accuracy the gate accepts (``deployed − ε``)."""
        return float(self.deployed_accuracy) - float(self.eps)

    def wall_clock_budget(self) -> float:
        """The wall-clock ceiling the gate accepts.

        An explicit ``wall_clock_budget_s`` wins; otherwise the frozen wall-clock
        scaled by ``(1 + wall_clock_slack)``.
        """
        if self.wall_clock_budget_s is not None:
            return float(self.wall_clock_budget_s)
        return float(self.wall_clock_s) * (1.0 + float(self.wall_clock_slack))

    def to_dict(self) -> Dict[str, Any]:
        # Unset F1 fields are omitted so a floor frozen without them round-trips byte-identically.
        data = asdict(self)
        for key in ("ac1_target", "ac2_reference", "ac5_budget_s"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegressionFloor":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"regression floor has unknown fields {sorted(unknown)} — the floor "
                f"format may have drifted; re-freeze the cell"
            )
        return cls(**dict(data))


@dataclass(frozen=True)
class CertificationFloorBook:
    """Immutable JSON-serializable book of frozen per-cell floors (``cell_key → RegressionFloor``)."""

    floors: Mapping[str, RegressionFloor] = field(default_factory=dict)
    format_version: int = FLOOR_BOOK_FORMAT_VERSION

    def floor_for(self, cell: CertificationCell) -> Optional[RegressionFloor]:
        return self.floors.get(cell.cell_key)

    def has_floor(self, cell: CertificationCell) -> bool:
        return cell.cell_key in self.floors

    def with_floor(
        self, cell: CertificationCell, floor: RegressionFloor
    ) -> "CertificationFloorBook":
        """Return a new book with ``cell``'s floor set/overwritten (immutable freeze)."""
        floors = dict(self.floors)
        floors[cell.cell_key] = floor
        return replace(self, floors=floors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "floors": {
                key: floor.to_dict() for key, floor in sorted(self.floors.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CertificationFloorBook":
        version = int(data.get("format_version", FLOOR_BOOK_FORMAT_VERSION))
        if version != FLOOR_BOOK_FORMAT_VERSION:
            raise ValueError(
                f"certification floor-book format_version {version} != "
                f"{FLOOR_BOOK_FORMAT_VERSION}; the format changed — migrate or "
                f"re-freeze the floor"
            )
        raw = data.get("floors", {})
        floors = {key: RegressionFloor.from_dict(val) for key, val in raw.items()}
        return cls(floors=floors, format_version=version)


class CertificationStatus(Enum):
    """The outcome of certifying a cell against its frozen floor."""

    PASS = "pass"
    FAIL = "fail"
    MISSING_FLOOR = "missing_floor"


@dataclass(frozen=True)
class AbsoluteVerdict:
    """The F1 absolute-AC sub-verdict — reported ALONGSIDE the relative gate.

    Each ``*_ok`` is ``None`` when its floor target is unset. ``accuracy_gap_pp`` is
    ``(deployed − ac1_target)`` in pp; ``ac5_gap_s`` is ``(max_ft_pass_wall_s − ac5_budget_s)`` in s.
    """

    ac1_ok: Optional[bool]
    ac2_ok: Optional[bool]
    ac5_ok: Optional[bool]
    accuracy_gap_pp: Optional[float]
    ac5_gap_s: Optional[float]


_ABSOLUTE_NONE = AbsoluteVerdict(None, None, None, None, None)


@dataclass(frozen=True)
class CertificationVerdict:
    """The per-cell gate result — PASS / FAIL / MISSING_FLOOR.

    ``accuracy_ok`` / ``wall_clock_ok`` decompose a FAIL by side; the ``absolute``
    F1 sub-verdict reports AC1/AC2/AC5 independent of the relative status.
    """

    cell_key: str
    status: CertificationStatus
    accuracy_ok: bool
    wall_clock_ok: bool
    measured_accuracy: float
    measured_wall_clock_s: float
    floor: Optional[RegressionFloor]
    reason: str
    absolute: AbsoluteVerdict = _ABSOLUTE_NONE

    @property
    def passed(self) -> bool:
        return self.status is CertificationStatus.PASS


def _absolute_verdict(
    floor: RegressionFloor,
    *,
    deployed_accuracy: float,
    max_ft_pass_wall_s: Optional[float],
) -> AbsoluteVerdict:
    """Compute the F1 absolute-AC sub-verdict (each axis None when its target unset)."""
    ac1_ok: Optional[bool] = None
    accuracy_gap_pp: Optional[float] = None
    if floor.ac1_target is not None:
        ac1_ok = deployed_accuracy >= float(floor.ac1_target)
        accuracy_gap_pp = (deployed_accuracy - float(floor.ac1_target)) * 100.0

    ac2_ok: Optional[bool] = None
    if floor.ac2_reference is not None:
        ac2_ok = deployed_accuracy >= float(floor.ac2_reference)

    ac5_ok: Optional[bool] = None
    ac5_gap_s: Optional[float] = None
    if floor.ac5_budget_s is not None and max_ft_pass_wall_s is not None:
        ac5_ok = float(max_ft_pass_wall_s) <= float(floor.ac5_budget_s)
        ac5_gap_s = float(max_ft_pass_wall_s) - float(floor.ac5_budget_s)

    return AbsoluteVerdict(
        ac1_ok=ac1_ok,
        ac2_ok=ac2_ok,
        ac5_ok=ac5_ok,
        accuracy_gap_pp=accuracy_gap_pp,
        ac5_gap_s=ac5_gap_s,
    )


def certify(
    cell: CertificationCell,
    *,
    deployed_accuracy: float,
    wall_clock_s: float,
    floor_book: CertificationFloorBook,
    max_ft_pass_wall_s: Optional[float] = None,
) -> CertificationVerdict:
    """Certify a cell's new numbers against its frozen floor.

    PASS iff ``deployed_accuracy >= floor.accuracy_floor()`` AND
    ``wall_clock_s <= floor.wall_clock_budget()``; a cell with no floor is MISSING_FLOOR
    (never a silent pass). The orthogonal ``absolute`` F1 sub-verdict never changes PASS/FAIL.
    """
    floor = floor_book.floor_for(cell)
    measured_accuracy = float(deployed_accuracy)
    measured_wall = float(wall_clock_s)

    if floor is None:
        return CertificationVerdict(
            cell_key=cell.cell_key,
            status=CertificationStatus.MISSING_FLOOR,
            accuracy_ok=False,
            wall_clock_ok=False,
            measured_accuracy=measured_accuracy,
            measured_wall_clock_s=measured_wall,
            floor=None,
            reason=(
                f"no frozen floor for cell {cell.cell_key!r} — freeze the matrix "
                f"before certifying (a missing floor is not a pass)"
            ),
            absolute=_ABSOLUTE_NONE,
        )

    absolute = _absolute_verdict(
        floor,
        deployed_accuracy=measured_accuracy,
        max_ft_pass_wall_s=max_ft_pass_wall_s,
    )

    accuracy_ok = measured_accuracy >= floor.accuracy_floor()
    wall_clock_ok = measured_wall <= floor.wall_clock_budget()

    if accuracy_ok and wall_clock_ok:
        status = CertificationStatus.PASS
        reason = (
            f"certified: accuracy {measured_accuracy:.4f} >= "
            f"{floor.accuracy_floor():.4f} and wall-clock {measured_wall:.1f}s <= "
            f"{floor.wall_clock_budget():.1f}s"
        )
    else:
        status = CertificationStatus.FAIL
        parts = []
        if not accuracy_ok:
            parts.append(
                f"accuracy {measured_accuracy:.4f} < floor {floor.accuracy_floor():.4f} "
                f"(deployed {floor.deployed_accuracy:.4f} − eps {floor.eps:.4f})"
            )
        if not wall_clock_ok:
            parts.append(
                f"wall-clock {measured_wall:.1f}s > budget "
                f"{floor.wall_clock_budget():.1f}s"
            )
        reason = "regression: " + "; ".join(parts)

    return CertificationVerdict(
        cell_key=cell.cell_key,
        status=status,
        accuracy_ok=accuracy_ok,
        wall_clock_ok=wall_clock_ok,
        measured_accuracy=measured_accuracy,
        measured_wall_clock_s=measured_wall,
        floor=floor,
        reason=reason,
        absolute=absolute,
    )


def freeze_cell(
    floor_book: CertificationFloorBook,
    cell: CertificationCell,
    *,
    deployed_accuracy: float,
    wall_clock_s: float,
    eps: float = DEFAULT_ACCURACY_EPS,
    wall_clock_slack: float = DEFAULT_WALL_CLOCK_SLACK,
    wall_clock_budget_s: Optional[float] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    ac1_target: Optional[float] = None,
    ac2_reference: Optional[float] = None,
    ac5_budget_s: Optional[float] = None,
) -> CertificationFloorBook:
    """Record the CURRENT deployed numbers for ``cell`` as its regression floor.

    Returns a NEW immutable :class:`CertificationFloorBook` with the cell frozen; the
    caller supplies the already-measured numbers. The optional F1 ``ac*`` targets
    default ``None`` ⇒ the frozen floor is byte-identical to the pre-overlay format.
    """
    floor = RegressionFloor(
        deployed_accuracy=float(deployed_accuracy),
        wall_clock_s=float(wall_clock_s),
        eps=float(eps),
        wall_clock_slack=float(wall_clock_slack),
        wall_clock_budget_s=(
            None if wall_clock_budget_s is None else float(wall_clock_budget_s)
        ),
        provenance=dict(provenance or {}),
        ac1_target=(None if ac1_target is None else float(ac1_target)),
        ac2_reference=(None if ac2_reference is None else float(ac2_reference)),
        ac5_budget_s=(None if ac5_budget_s is None else float(ac5_budget_s)),
    )
    return floor_book.with_floor(cell, floor)


def floor_is_stale(
    floor: RegressionFloor, controller_commit: Optional[str]
) -> bool:
    """True iff the controller-path commit moved since the floor froze (a warn, not a hard fail).

    A floor with no recorded commit is stale (cannot prove freshness);
    ``controller_commit is None`` (nothing to compare) never flags.
    """
    if controller_commit is None:
        return False
    frozen_commit = floor.provenance.get("commit") if floor.provenance else None
    if not frozen_commit:
        return True
    return str(frozen_commit) != str(controller_commit)


def load_floor_book(path: str) -> CertificationFloorBook:
    """Load a frozen floor book from a JSON file (the format a matrix run wrote)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return CertificationFloorBook.from_dict(data)


def save_floor_book(floor_book: CertificationFloorBook, path: str) -> None:
    """Persist a floor book to a JSON file (stable key order for a clean diff)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(floor_book.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
