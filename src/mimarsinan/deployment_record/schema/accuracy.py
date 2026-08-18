"""Deployed accuracy reads, faithfulness certificates, and adaptation walls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.deployment_record.schema.serde import (
    require_choice,
    strict_kwargs,
    tuple_of,
)

READ_BACKENDS = frozenset({"hcm", "nevresim", "value_census", "pipeline"})
READ_KINDS = frozenset({"measured", "carried"})


@dataclass(frozen=True)
class AccuracyReadRecord:
    """One accuracy read: which backend produced it, at which step, on how many samples."""

    metric: float
    backend: str
    samples: int
    kind: str
    step: str

    def __post_init__(self) -> None:
        require_choice("AccuracyReadRecord", "backend", self.backend, READ_BACKENDS)
        require_choice("AccuracyReadRecord", "kind", self.kind, READ_KINDS)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccuracyReadRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class CertificateRecord:
    """One faithfulness certificate outcome (the FATAL gates behind 'certified')."""

    name: str
    backend: str
    passed: bool
    neuron_windows_compared: int
    exact_match_fraction: float
    max_abs_delta: float
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CertificateRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class AccuracyRecord:
    """The accuracy fragment: 'certified' = deployed read + FATAL gates green."""

    deployed: AccuracyReadRecord
    reads: Tuple[AccuracyReadRecord, ...]
    certificates: Tuple[CertificateRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccuracyRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["deployed"] = AccuracyReadRecord.from_dict(kwargs["deployed"])
        kwargs["reads"] = tuple_of(AccuracyReadRecord.from_dict, kwargs["reads"])
        kwargs["certificates"] = tuple_of(
            CertificateRecord.from_dict, kwargs["certificates"]
        )
        return cls(**kwargs)


@dataclass(frozen=True)
class FtPassWallRecord:
    """One fine-tuning pass wall: ``{label, wall_s}``."""

    label: str
    wall_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FtPassWallRecord":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class AdaptationRecord:
    """The adaptation fragment: per-fine-tuning-pass wall bundles."""

    max_ft_pass_wall_s: float
    ft_pass_walls: Tuple[FtPassWallRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdaptationRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["ft_pass_walls"] = tuple_of(
            FtPassWallRecord.from_dict, kwargs["ft_pass_walls"]
        )
        return cls(**kwargs)


@dataclass(frozen=True)
class AdaptationSummaryRecord:
    """[TS5] What the adaptation controllers spent and how each run ended.

    Totals only: the per-event trace stays in the steps' own
    ``<Step>.adaptation_ledger.json`` artifacts. ``stalls_by_path`` counts the
    escalations the run took; ``completed_via`` names, per adaptation step, the
    path its rate search exited through.
    """

    proposed: int
    accepted: int
    rejected: int
    retries: int
    recovery_steps: int
    probe_evals: int
    endpoint_steps: int
    total_steps: int
    stalls_by_path: Mapping[str, int]
    completed_via: Mapping[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdaptationSummaryRecord":
        kwargs = strict_kwargs(cls, data)
        kwargs["stalls_by_path"] = {
            str(k): int(v) for k, v in dict(kwargs["stalls_by_path"]).items()
        }
        kwargs["completed_via"] = {
            str(k): str(v) for k, v in dict(kwargs["completed_via"]).items()
        }
        return cls(**kwargs)
