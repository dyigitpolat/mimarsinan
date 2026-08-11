"""The sealed artifact: identity + fragments, versioned, unknown fields rejected."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from mimarsinan.deployment_record.schema.accuracy import AccuracyRecord, AdaptationRecord
from mimarsinan.deployment_record.schema.physics import EnergyRecord, TimingRecord
from mimarsinan.deployment_record.schema.placement import PlacementRecord
from mimarsinan.deployment_record.schema.provenance import Provenance
from mimarsinan.deployment_record.schema.schedule import ScheduleRecord
from mimarsinan.deployment_record.schema.serde import optional, strict_kwargs
from mimarsinan.deployment_record.schema.traffic import TrafficRecord
from mimarsinan.deployment_record.schema.utilization import UtilizationRecord

DEPLOYMENT_RECORD_FORMAT_VERSION = 1
DEPLOYMENT_RECORD_FILENAME = "deployment_record.json"


def _require_version(cls_name: str, version: Any) -> None:
    if int(version) != DEPLOYMENT_RECORD_FORMAT_VERSION:
        raise ValueError(
            f"{cls_name} format_version {version} != "
            f"{DEPLOYMENT_RECORD_FORMAT_VERSION}; the format changed — "
            f"migrate explicitly, never tolerate silently"
        )


@dataclass(frozen=True)
class RecordIdentity:
    """What was deployed, where, under which resolved options — the record's key."""

    format_version: int
    run_dir: str
    cell_key: str
    mode: str
    model_type: str
    model_name: str
    workload: str
    config_digest: str
    platform: Mapping[str, Any]
    deployment_options: Mapping[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        _require_version("RecordIdentity", self.format_version)

    def to_dict(self) -> Dict[str, Any]:
        data = {f: getattr(self, f) for f in self.__dataclass_fields__}
        data["platform"] = dict(self.platform)
        data["deployment_options"] = dict(self.deployment_options)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordIdentity":
        _require_version(cls.__name__, dict(data).get("format_version", -1))
        kwargs = strict_kwargs(cls, data)
        kwargs["platform"] = dict(kwargs["platform"])
        kwargs["deployment_options"] = dict(kwargs["deployment_options"])
        return cls(**kwargs)


@dataclass(frozen=True)
class DeploymentRecord:
    """The one sealed deployment artifact — a join of the run's fragments."""

    identity: RecordIdentity
    schedule: ScheduleRecord
    placement: PlacementRecord
    utilization: UtilizationRecord
    accuracy: AccuracyRecord
    timing: TimingRecord
    traffic: Optional[TrafficRecord]
    energy: Optional[EnergyRecord]
    adaptation: Optional[AdaptationRecord]
    provenance: Mapping[str, Provenance] = field(default_factory=dict)
    format_version: int = DEPLOYMENT_RECORD_FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_version("DeploymentRecord", self.format_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "schedule": self.schedule.to_dict(),
            "placement": self.placement.to_dict(),
            "utilization": self.utilization.to_dict(),
            "accuracy": self.accuracy.to_dict(),
            "timing": self.timing.to_dict(),
            "traffic": None if self.traffic is None else self.traffic.to_dict(),
            "energy": None if self.energy is None else self.energy.to_dict(),
            "adaptation": None if self.adaptation is None else self.adaptation.to_dict(),
            "provenance": {k: v.to_dict() for k, v in self.provenance.items()},
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeploymentRecord":
        _require_version(
            cls.__name__,
            dict(data).get("format_version", DEPLOYMENT_RECORD_FORMAT_VERSION),
        )
        kwargs = strict_kwargs(cls, data)
        kwargs["identity"] = RecordIdentity.from_dict(kwargs["identity"])
        kwargs["schedule"] = ScheduleRecord.from_dict(kwargs["schedule"])
        kwargs["placement"] = PlacementRecord.from_dict(kwargs["placement"])
        kwargs["utilization"] = UtilizationRecord.from_dict(kwargs["utilization"])
        kwargs["accuracy"] = AccuracyRecord.from_dict(kwargs["accuracy"])
        kwargs["timing"] = TimingRecord.from_dict(kwargs["timing"])
        kwargs["traffic"] = optional(TrafficRecord.from_dict, kwargs["traffic"])
        kwargs["energy"] = optional(EnergyRecord.from_dict, kwargs["energy"])
        kwargs["adaptation"] = optional(AdaptationRecord.from_dict, kwargs["adaptation"])
        kwargs["provenance"] = {
            key: Provenance.from_dict(value)
            for key, value in dict(kwargs["provenance"]).items()
        }
        return cls(**kwargs)


def save_deployment_record(record: DeploymentRecord, run_dir: str) -> str:
    """Atomically write ``deployment_record.json`` into ``run_dir``; return the path."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, DEPLOYMENT_RECORD_FILENAME)
    fd, tmp_path = tempfile.mkstemp(
        dir=run_dir, prefix=DEPLOYMENT_RECORD_FILENAME, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return path


def load_deployment_record(path: str) -> DeploymentRecord:
    """Load a sealed record from a JSON file (the format a run wrote)."""
    with open(path, "r", encoding="utf-8") as fh:
        return DeploymentRecord.from_dict(json.load(fh))
