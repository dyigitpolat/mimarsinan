"""Structured decision-trace artifact for adaptation cycles (golden-trace source)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Iterator, Optional

_LEGACY_KEYS = {
    "catastrophic": ("rate", "committed", "instant_acc", "outcome", "elapsed_sec"),
    "rollback": (
        "rate", "committed", "instant_acc", "pre_cycle_acc",
        "post_acc", "lr", "outcome", "elapsed_sec",
    ),
    "commit": (
        "rate", "committed", "pre_cycle_acc", "post_acc",
        "lr", "reached_target", "outcome", "elapsed_sec",
    ),
}


@dataclass(frozen=True)
class DecisionRecord:
    """One adaptation-cycle decision. Fields absent at an exit stay ``None``."""

    cycle_index: int
    outcome: str
    rate: float
    committed: float
    elapsed_sec: float = 0.0
    instant_acc: Optional[float] = None
    pre_cycle_acc: Optional[float] = None
    post_acc: Optional[float] = None
    lr: Optional[float] = None
    reached_target: Optional[bool] = None
    target: Optional[float] = None
    validation_baseline: Optional[float] = None
    rollback_threshold: Optional[float] = None
    absolute_floor: Optional[float] = None
    rollback_tolerance: Optional[float] = None
    seeds: Optional[dict] = None

    def as_legacy_dict(self) -> dict:
        """The per-outcome ``_cycle_log`` dict for this record.

        ``rate`` and ``committed`` are always numeric — ``_log_cycle_summary``
        applies ``:.4f`` and would crash on a missing value.
        """
        keys = _LEGACY_KEYS.get(self.outcome, ("rate", "committed", "outcome", "elapsed_sec"))
        out = {k: getattr(self, k) for k in keys}
        out["rate"] = float(self.rate)
        out["committed"] = float(self.committed)
        return out


class DecisionTrace:
    """Ordered, JSON-round-trippable stream of :class:`DecisionRecord`.

    Iterates as legacy per-outcome dicts so ``_log_cycle_summary`` consumes it unchanged.
    """

    def __init__(self, records: Optional[list] = None):
        self._records: list[DecisionRecord] = list(records) if records else []

    @classmethod
    def new(cls) -> "DecisionTrace":
        return cls()

    def record(self, rec: DecisionRecord) -> None:
        self._records.append(rec)

    @property
    def records(self) -> tuple:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)

    def __iter__(self) -> Iterator[dict]:
        for rec in self._records:
            yield rec.as_legacy_dict()

    def __getitem__(self, index: int) -> dict:
        return self._records[index].as_legacy_dict()

    def to_json(self) -> str:
        """Deterministic serialization (sorted keys) — the golden artifact."""
        return json.dumps(
            [dataclasses.asdict(rec) for rec in self._records],
            sort_keys=True,
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "DecisionTrace":
        return cls([DecisionRecord(**entry) for entry in json.loads(text)])
