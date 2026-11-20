"""The sealed artifact: identity + fragments, versioned, unknown fields rejected."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.common.refusal.format_version import FormatVersionError, exact_version
from mimarsinan.deployment_record.schema.accuracy import (
    AccuracyRecord,
    AdaptationRecord,
    AdaptationSummaryRecord,
)
from mimarsinan.deployment_record.schema.physics import EnergyRecord, TimingRecord
from mimarsinan.deployment_record.schema.placement import PlacementRecord
from mimarsinan.deployment_record.schema.provenance import Provenance
from mimarsinan.deployment_record.schema.schedule import ScheduleRecord
from mimarsinan.deployment_record.schema.serde import optional, strict_kwargs
from mimarsinan.deployment_record.schema.traffic import TrafficRecord
from mimarsinan.deployment_record.schema.utilization import UtilizationRecord

#: [composed head] Format 3 holds both named changes since the base: [U-HOST2] the sealed raw host wall named apart
#: from the per-inference quantity (timing.latency's host_ops_s_raw_total, the per-pass wall per inference) and
#: [U-LIF2 W7] the window firing in the identity with each pass's dependency depth. A format-1 record is migrated
#: explicitly on read (``migrate_format_1``: ``VERSION_1_RULE``, then the window firing's rule); a format-2 record is
#: refused by name (each unit sealed its own format 2 on its own tree, never a record of record).
DEPLOYMENT_RECORD_FORMAT_VERSION = 3
DEPLOYMENT_RECORD_FILENAME = "deployment_record.json"
#: [U-HOST2] The one stated rule a version 1 record is read by.
VERSION_1_RULE = ("a version 1 record is read as version 3: timing.latency.host_ops_s (the raw total over the whole "
                  "run) becomes host_ops_s_raw_total with its value, host_ops_s_per_pass (per call in version 1, never "
                  "per inference) is withheld as None, and every other field is read as sealed")


#: [U-HOST3] The fields ``VERSION_1_RULE`` reads, each required of a version 1 record (missing, it refuses by name).
VERSION_1_FIELDS = (("identity", "format_version"), ("timing", "latency", "host_ops_s"),
                    ("timing", "latency", "host_ops_s_per_pass"))


class RecordVersionError(FormatVersionError):
    """[U-HOST3] A record of another version than the one read, or a version 1 record without a field its rule reads,
    refused by name (the version itself read by ``common.refusal.format_version.exact_version``)."""


def from_version_1(data: Mapping[str, Any]) -> Dict[str, Any]:
    """``VERSION_1_RULE`` applied to a version 1 record's dict (a copy; the input is not changed). [U-HOST3] Every field
    the rule reads is required (``VERSION_1_FIELDS``), its identity states version 1, and it does not already name
    the version 2 field; else it refuses by name."""
    out = json.loads(json.dumps(dict(data)))
    for path in VERSION_1_FIELDS:
        holder: Any = out
        for depth, part in enumerate(path):
            if not isinstance(holder, Mapping) or part not in holder:
                raise RecordVersionError(f"a version 1 record without {'.'.join(path[:depth + 1])}: VERSION_1_RULE "
                                         f"reads {['.'.join(p) for p in VERSION_1_FIELDS]}")
            holder = holder[part]
    if exact_version(out["identity"]["format_version"], "a version 1 record's identity") != 1:
        raise RecordVersionError(f"a version 1 record's identity states format_version "
                                 f"{out['identity']['format_version']!r}: one record has one version")
    latency = out["timing"]["latency"]
    if "host_ops_s_raw_total" in latency:
        raise RecordVersionError("a version 1 record already names host_ops_s_raw_total, the version 2 field "
                                 "VERSION_1_RULE writes: it is no version 1 record")
    out["format_version"] = DEPLOYMENT_RECORD_FORMAT_VERSION
    out["identity"]["format_version"] = DEPLOYMENT_RECORD_FORMAT_VERSION
    latency["host_ops_s_raw_total"] = latency.pop("host_ops_s")
    latency["host_ops_s_per_pass"] = None
    return out


def migrate_format_1(data: Mapping[str, Any]) -> Dict[str, Any]:
    """A format-1 record dict as the current format: ``VERSION_1_RULE`` first (the raw host wall named apart, both
    versions written), then its identity states no window firing and its passes no dependency depth, which readers
    take as what such a record ran (firing while integrating, one level a pass). Any other version is returned as it
    is, for the version check to judge."""
    record = dict(data)
    if exact_version(record.get("format_version", DEPLOYMENT_RECORD_FORMAT_VERSION), "DeploymentRecord") != 1:
        return record
    record = from_version_1(record)
    record["identity"].setdefault("window_firing", None)
    return record


def _require_version(cls_name: str, version: Any) -> None:
    if exact_version(version, cls_name) != DEPLOYMENT_RECORD_FORMAT_VERSION:
        raise RecordVersionError(
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
    #: [U-LIF2 W7] The run's resolved lif_window_firing (``None`` away from the synchronized LIF variant).
    window_firing: Optional[str] = None

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
    # [TS5] additive-optional (the ``invocations`` / ``partition`` precedent):
    # LAST, defaults to None, no format-version bump — a record written before
    # the controller ledger existed still loads.
    adaptation_ledger: Optional[AdaptationSummaryRecord] = None
    #: [W08] The output stage of a chip declaring ``onchip_arithmetic`` (``mapping.lanes.output_stage``): lane
    #: work, host boundary bytes and round trips, labelled executed or estimated. Emitted only when set, so every
    #: record of a chip without the declaration is byte-identical.
    output_stage: Optional[Mapping[str, Any]] = None
    #: [U-C C-5] Each bundle key the run's ``target_overrides`` replaces (key, bundle value, declared value, note),
    #: printed by name; emitted only when the run departs from its target bundle, so every other record is
    #: byte-identical.
    declared_departures: Optional[Tuple[Mapping[str, Any], ...]] = None
    #: [U-HV] Each ODIN step's sample read (backend, the declaring key, the split, the per-split indices); emitted
    #: only when such a step ran, so every other record is byte-identical.
    device_samples: Optional[Tuple[Mapping[str, Any], ...]] = None
    #: [U-BRANGE F-2] Boundary Quantization's report: its range rule and, per boundary, the scale and the clamp census
    #: on the calibration batches (clamped fraction, the clamped values' share of the squared magnitude, the largest
    #: value over the scale); emitted only when the run grids its boundaries, so every other record is byte-identical.
    boundary_ranges: Optional[Mapping[str, Any]] = None

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
            "adaptation_ledger": (
                None if self.adaptation_ledger is None
                else self.adaptation_ledger.to_dict()
            ),
            **({} if self.output_stage is None else {"output_stage": json.loads(json.dumps(self.output_stage))}),
            **({} if not self.declared_departures else {
                "declared_departures": json.loads(json.dumps(list(self.declared_departures)))}),
            **({} if not self.device_samples else {
                "device_samples": json.loads(json.dumps(list(self.device_samples)))}),
            **({} if self.boundary_ranges is None else {
                "boundary_ranges": json.loads(json.dumps(self.boundary_ranges))}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeploymentRecord":
        if exact_version(dict(data).get("format_version", DEPLOYMENT_RECORD_FORMAT_VERSION), cls.__name__) == 1:
            data = migrate_format_1(data)  # [U-HOST3] exact versions; VERSION_1_RULE, then the window firing's rule
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
        # Additive-optional: an absent key keeps the field's None default.
        if "adaptation_ledger" in kwargs:
            kwargs["adaptation_ledger"] = optional(
                AdaptationSummaryRecord.from_dict, kwargs["adaptation_ledger"]
            )
        if kwargs.get("output_stage") is not None:
            kwargs["output_stage"] = dict(kwargs["output_stage"])
        if kwargs.get("declared_departures") is not None:
            kwargs["declared_departures"] = tuple(dict(d) for d in kwargs["declared_departures"])
        if kwargs.get("device_samples") is not None:
            kwargs["device_samples"] = tuple(dict(d) for d in kwargs["device_samples"])
        if kwargs.get("boundary_ranges") is not None:
            kwargs["boundary_ranges"] = dict(kwargs["boundary_ranges"])
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
