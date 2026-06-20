"""Certification protocol (Frontier E6): the mechanism that REPLACES byte-identity
as the Fix-B gate.

Fix B breaks every equivalence lock *by design* — it changes the deployed numbers.
This module is the protocol the review (C3) and the action plan (Fix B requirement
#1, "the protocol you do not yet have") call for and the refactor did not build: a
frozen per-cell **regression floor** and a per-cell **Pareto/regression gate** on the
deployed-forward metric (R6 / E5) plus the wall-clock budget.

It declares and computes; it does **not** run the matrix. The floor is populated by a
GPU matrix run *before the first Fix-B flip* (see ``docs/CERTIFICATION_PROTOCOL.md``);
this module is the format + freezing + gating mechanism only — pure additive infra,
byte-identical (no existing deployment path reads it yet).

Three pieces:

* :class:`CertificationCell` — the matrix coordinate the gate is keyed by:
  ``(firing × sync × backend)``. ``cell_key`` is the canonical string (it reuses the
  ``mode[/schedule]`` naming the E4 proposer / E3 calibration already use, suffixed
  ``@backend``) so a cell names the same thing across the whole program.
* :class:`RegressionFloor` — the frozen ``{deployed_accuracy, wall_clock_s}`` per cell,
  with the budget/epsilon the gate compares against. :class:`CertificationFloorBook`
  is the JSON-serializable book of floors (the FORMAT a matrix run freezes), with
  :func:`freeze_cell` recording the CURRENT numbers for a cell.
* :func:`certify` — the GATE: given a cell's new ``(deployed_accuracy, wall_clock_s)``
  vs the frozen floor, PASS iff ``deployed_accuracy >= floor − ε`` **and**
  ``wall_clock_s <= budget``. A cell with no frozen floor is :class:`CertificationStatus`
  ``MISSING_FLOOR`` (cannot certify — freeze the floor first), never a silent pass.
"""

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
    "DEFAULT_ACCURACY_EPS",
    "DEFAULT_WALL_CLOCK_SLACK",
    "freeze_cell",
    "certify",
    "load_floor_book",
    "save_floor_book",
    "FLOOR_BOOK_FORMAT_VERSION",
]


# The frozen-floor FORMAT version. A matrix run writes this into the JSON so a
# future format change fails loud instead of silently mis-reading an old book.
FLOOR_BOOK_FORMAT_VERSION = 1

# The certification tolerances. ``eps`` is the deployed-accuracy slack the Fix-B
# action plan calls for (``deployed >= floor − ε``); ``wall_clock`` budgets the
# speed side (``wall_clock <= floor.wall_clock_s × (1 + slack)`` unless an absolute
# budget is set on the floor). Defaults are conservative placeholders the freezing
# matrix run / config overrides; they are NOT a behavior change (nothing reads them
# until Fix B wires the gate).
DEFAULT_ACCURACY_EPS = 0.0
DEFAULT_WALL_CLOCK_SLACK = 0.0


@dataclass(frozen=True)
class CertificationCell:
    """The matrix coordinate the per-cell gate is keyed by: (firing × sync × backend).

    ``firing`` is the spiking mode (``lif`` / ``rate`` / ``ttfs`` / ``ttfs_quantized``
    / ``ttfs_cycle_based``); ``sync`` is the ``ttfs_cycle_schedule``
    (``cascaded`` / ``synchronized``) or ``None`` for the non-cycle families;
    ``backend`` is the deployment simulator the number was measured on (``nevresim``
    / ``sanafe`` / ``lava`` / ``hcm`` …). The triple is exactly the cell granularity
    Fix B rolls out per (lowest-risk-first), so the floor is keyed by it.
    """

    firing: str
    sync: Optional[str]
    backend: str

    @property
    def cell_key(self) -> str:
        """Canonical string key — ``mode[/schedule]@backend``.

        Reuses the ``mode[/schedule]`` naming the E4 proposer and E3 calibration
        already use (so a cell is the SAME named thing across the program), suffixed
        with the backend it was measured on.
        """
        mode = self.firing if self.sync is None else f"{self.firing}/{self.sync}"
        return f"{mode}@{self.backend}"

    @classmethod
    def from_key(cls, key: str) -> "CertificationCell":
        """Parse a canonical ``mode[/schedule]@backend`` key back into a cell."""
        if "@" not in key:
            raise ValueError(
                f"certification cell key {key!r} is missing the '@backend' suffix"
            )
        mode_part, backend = key.rsplit("@", 1)
        if "/" in mode_part:
            firing, sync = mode_part.split("/", 1)
        else:
            firing, sync = mode_part, None
        if not firing or not backend:
            raise ValueError(f"certification cell key {key!r} is malformed")
        return cls(firing=firing, sync=sync, backend=backend)

    @classmethod
    def from_mode_policy(cls, mode_policy: Any, *, backend: str) -> "CertificationCell":
        """Build a cell from a (firing × sync) ``SpikingModePolicy`` + a backend.

        Reuses the policy's ``spiking_mode`` / ``schedule`` so the cell names the
        same (firing × sync) thing the E3/E4 layers key on.
        """
        firing = str(getattr(mode_policy, "spiking_mode", "lif"))
        sync = getattr(mode_policy, "schedule", None)
        sync = None if sync is None else str(sync)
        return cls(firing=firing, sync=sync, backend=str(backend))


@dataclass(frozen=True)
class RegressionFloor:
    """The frozen deployed floor for one cell — the regression baseline the gate uses.

    ``deployed_accuracy`` is the deployed-forward, full-test, parity-gated number
    (the only number of record per R6 / E5) the new run must not regress below by
    more than ``eps``. ``wall_clock_s`` is the measured per-step wall-clock the new
    run must not exceed (subject to ``wall_clock_slack`` or an absolute
    ``wall_clock_budget_s`` override). ``eps`` / ``wall_clock_slack`` /
    ``wall_clock_budget_s`` are the per-cell tolerances frozen alongside the numbers
    so the gate is fully self-describing. ``provenance`` records how/when the floor
    was frozen (commit, date, sample count) for the "this commit changes numbers,
    here is the new certified baseline" story.
    """

    deployed_accuracy: float
    wall_clock_s: float
    eps: float = DEFAULT_ACCURACY_EPS
    wall_clock_slack: float = DEFAULT_WALL_CLOCK_SLACK
    wall_clock_budget_s: Optional[float] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

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
        return asdict(self)

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
    """The JSON-serializable book of frozen per-cell floors (the FROZEN FORMAT).

    A mapping ``cell_key → RegressionFloor`` plus a format version. A matrix run
    freezes one book (one floor per (firing × sync × backend) cell it measured); CI
    loads it and gates each new run against the matching cell. The book is immutable;
    :meth:`with_floor` returns a new book with one cell frozen (so freezing is a
    pure, auditable operation).
    """

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
class CertificationVerdict:
    """The per-cell gate result — the certified/rejected/uncertifiable verdict.

    ``status`` is the trichotomy: PASS (certified), FAIL (regressed accuracy or blew
    the budget — the gate rejects the change), MISSING_FLOOR (no frozen baseline for
    this cell — it cannot be certified until the floor is frozen, NEVER a silent
    pass). ``accuracy_ok`` / ``wall_clock_ok`` decompose a FAIL so the caller can
    report WHICH side regressed; ``reason`` is the human-facing explanation.
    """

    cell_key: str
    status: CertificationStatus
    accuracy_ok: bool
    wall_clock_ok: bool
    measured_accuracy: float
    measured_wall_clock_s: float
    floor: Optional[RegressionFloor]
    reason: str

    @property
    def passed(self) -> bool:
        return self.status is CertificationStatus.PASS


def certify(
    cell: CertificationCell,
    *,
    deployed_accuracy: float,
    wall_clock_s: float,
    floor_book: CertificationFloorBook,
) -> CertificationVerdict:
    """The Fix-B GATE: certify a cell's new numbers against its frozen floor.

    PASS iff ``deployed_accuracy >= floor.accuracy_floor()`` (``floor − ε``) **and**
    ``wall_clock_s <= floor.wall_clock_budget()``. A FAIL names which side regressed.
    A cell with no frozen floor is ``MISSING_FLOOR`` — it cannot be certified (freeze
    the floor first via :func:`freeze_cell`), so the gate never passes a cell it has
    no baseline for. This replaces the byte-identity equivalence lock for Fix B.
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
) -> CertificationFloorBook:
    """Record the CURRENT deployed numbers for ``cell`` as its regression floor.

    Returns a NEW :class:`CertificationFloorBook` with the cell frozen (the book is
    immutable). This is what the matrix-run freezing script calls per cell to record
    the current slow/lossy baseline BEFORE the first Fix-B flip. Pure data — it never
    runs the matrix; the caller supplies the already-measured deployed-metric numbers.
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
    )
    return floor_book.with_floor(cell, floor)


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
