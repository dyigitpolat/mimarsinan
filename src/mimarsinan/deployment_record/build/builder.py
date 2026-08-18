"""Attach-once record assembly, sealed against the plan-derived required-fragment matrix."""

from __future__ import annotations

from typing import AbstractSet, Any, Dict, Optional, Protocol, Sequence

from mimarsinan.deployment_record.schema import (
    AccuracyRecord,
    AdaptationRecord,
    AdaptationSummaryRecord,
    DeploymentRecord,
    EnergyRecord,
    PlacementRecord,
    Provenance,
    RecordIdentity,
    ScheduleRecord,
    TimingRecord,
    TrafficRecord,
    UtilizationRecord,
)

_FRAGMENT_TYPES: Dict[str, type] = {
    "identity": RecordIdentity,
    "schedule": ScheduleRecord,
    "placement": PlacementRecord,
    "utilization": UtilizationRecord,
    "accuracy": AccuracyRecord,
    "timing": TimingRecord,
    "traffic": TrafficRecord,
    "energy": EnergyRecord,
    "adaptation": AdaptationRecord,
    "adaptation_ledger": AdaptationSummaryRecord,
}

_ALWAYS_REQUIRED = (
    "identity", "schedule", "placement", "utilization", "accuracy", "timing",
)


class SealPlanView(Protocol):
    """The duck surface ``seal`` consumes so this module never imports ``pipelining``.

    An adapter over the resolved ``DeploymentPlan`` provides: the SANA-FE enable,
    the counts-observable mode predicate, the spike-count-gate arming, nevresim
    applicability, and the names of tuner-hosting adaptation steps (matched
    against ``resolved_step_names``).
    """

    enable_sanafe_simulation: bool
    counts_observable: bool
    spike_count_gate_armed: bool
    nevresim_applies: bool
    tuner_hosting_step_names: AbstractSet[str]


class DeploymentRecordBuilder:
    """Collects fragment groups exactly once each, then seals them into a record.

    Every ``attach`` names a group from the fragment table; a second attach of
    the same group raises. ``seal`` enforces the spec's required-fragment matrix
    (docs/deployment_record_schema.md §5), the cross-checks between the three
    formerly-disconnected surfaces, and one provenance entry per attached group.
    """

    def __init__(self) -> None:
        self._fragments: Dict[str, Any] = {}
        self._provenance: Dict[str, Provenance] = {}
        self._weight_programming_params: Optional[int] = None

    def attach(
        self, group: str, fragment: Any, provenance: Optional[Provenance] = None
    ) -> None:
        """Attach one fragment group (exactly once), with its provenance entry."""
        expected = _FRAGMENT_TYPES.get(group)
        if expected is None:
            raise ValueError(
                f"unknown fragment group {group!r}; expected one of "
                f"{sorted(_FRAGMENT_TYPES)}"
            )
        if group in self._fragments:
            raise ValueError(f"fragment group {group!r} attached twice")
        if not isinstance(fragment, expected):
            raise TypeError(
                f"fragment group {group!r} requires {expected.__name__}, "
                f"got {type(fragment).__name__}"
            )
        self._fragments[group] = fragment
        if provenance is not None:
            self._provenance[group] = provenance

    def declare_weight_programming_totals(self, *, params_programmed: int) -> None:
        """Declare the independent weight-programming-report total (exactly once)."""
        if self._weight_programming_params is not None:
            raise ValueError("weight-programming totals declared twice")
        self._weight_programming_params = int(params_programmed)

    def seal(
        self, plan: SealPlanView, resolved_step_names: Sequence[str]
    ) -> DeploymentRecord:
        """Validate the required matrix + cross-checks and return the sealed record."""
        self._require_matrix(plan, resolved_step_names)
        self._require_provenance()
        self._cross_check()
        return DeploymentRecord(
            identity=self._fragments["identity"],
            schedule=self._fragments["schedule"],
            placement=self._fragments["placement"],
            utilization=self._fragments["utilization"],
            accuracy=self._fragments["accuracy"],
            timing=self._fragments["timing"],
            traffic=self._fragments.get("traffic"),
            energy=self._fragments.get("energy"),
            adaptation=self._fragments.get("adaptation"),
            provenance=dict(self._provenance),
            adaptation_ledger=self._fragments.get("adaptation_ledger"),
        )

    def _require(self, group: str, reason: str) -> Any:
        if group not in self._fragments:
            raise ValueError(f"seal: required fragment {group!r} missing ({reason})")
        return self._fragments[group]

    def _require_matrix(
        self, plan: SealPlanView, resolved_step_names: Sequence[str]
    ) -> None:
        for group in _ALWAYS_REQUIRED:
            self._require(group, "always required")
        if plan.enable_sanafe_simulation:
            self._require("energy", "enable_sanafe_simulation")
            timing: TimingRecord = self._fragments["timing"]
            if not timing.per_segment:
                raise ValueError(
                    "seal: timing.per_segment empty (enable_sanafe_simulation)"
                )
            traffic: TrafficRecord = self._require("traffic", "enable_sanafe_simulation")
            if traffic.noc is None:
                raise ValueError("seal: traffic.noc missing (enable_sanafe_simulation)")
            placement: PlacementRecord = self._fragments["placement"]
            if placement.floorplan is None:
                raise ValueError(
                    "seal: placement.floorplan missing (enable_sanafe_simulation)"
                )
        if plan.counts_observable and plan.spike_count_gate_armed:
            traffic = self._require(
                "traffic", "counts-observable mode with the spike-count gate armed"
            )
            if traffic.boundaries is None:
                raise ValueError(
                    "seal: traffic.boundaries missing (counts-observable mode "
                    "with the spike-count gate armed)"
                )
        if plan.nevresim_applies:
            accuracy: AccuracyRecord = self._fragments["accuracy"]
            if not any(read.backend == "nevresim" for read in accuracy.reads):
                raise ValueError(
                    "seal: accuracy.reads has no 'nevresim' read (nevresim applies)"
                )
        if set(resolved_step_names) & set(plan.tuner_hosting_step_names):
            self._require("adaptation", "tuner-hosting adaptation step resolved")

    def _require_provenance(self) -> None:
        missing = sorted(set(self._fragments) - set(self._provenance))
        if missing:
            raise ValueError(
                f"seal: fragment groups {missing} have no provenance entry"
            )

    def _cross_check(self) -> None:
        schedule: ScheduleRecord = self._fragments["schedule"]
        utilization: UtilizationRecord = self._fragments["utilization"]
        segments = schedule.segments()
        if schedule.pass_count != utilization.layout.schedule_pass_count:
            raise ValueError(
                f"seal cross-check pass_count: schedule.pass_count "
                f"{schedule.pass_count} != utilization.layout.schedule_pass_count "
                f"{utilization.layout.schedule_pass_count}"
            )
        if self._weight_programming_params is None:
            raise ValueError(
                "seal cross-check params_programmed: weight-programming totals "
                "never declared (declare_weight_programming_totals)"
            )
        scheduled_params = sum(seg.params_programmed for seg in segments)
        if scheduled_params != self._weight_programming_params:
            raise ValueError(
                f"seal cross-check params_programmed: sum over segments "
                f"{scheduled_params} != weight-programming report total "
                f"{self._weight_programming_params}"
            )
        scheduled_cores = sum(len(seg.cores) for seg in segments)
        if utilization.crossbar.cores_allocated != scheduled_cores:
            raise ValueError(
                f"seal cross-check cores_allocated: utilization.crossbar."
                f"cores_allocated {utilization.crossbar.cores_allocated} != "
                f"sum of segment cores {scheduled_cores}"
            )
