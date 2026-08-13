"""Backings: one generic form for "which datum answers this objective, and is it there".

Every objective is declared as a :class:`Backing` — a single reader that returns
the value or ``None`` when its datum is absent. Availability and extraction are
then the SAME function, so an objective can never claim to be available and then
fail to produce a number (or vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from mimarsinan.deployment_record.cost.terms import CostTerm, find_term_or_none
from mimarsinan.deployment_record.objectives.spec import RecordView

Reader = Callable[[RecordView], Optional[float]]

_LAYOUT_REQUIRES = "layout verification statistics (the mapping's packing census)"
_SYNC_REQUIRES = (
    "layout verification statistics AND the host-side segment census (a search "
    "candidate's layout view)"
)
_SCHEDULE_REQUIRES = "the sealed record's schedule fragment (the pass census)"
_ENERGY_REQUIRES = "the sealed record's energy fragment (SANA-FE measured)"
_TIMING_REQUIRES = "the measured per-segment timing census (SANA-FE)"
_NOC_REQUIRES = "the sealed record's NoC traffic fragment (SANA-FE)"
_PHYSICS_REQUIRES = (
    "declared platform physics (platform_physics_profile / overrides) whose "
    "constants can back this axis, plus the quantities it prices"
)
_COST_REQUIRES = (
    "the cost model's inputs on a sealed record: the energy fragment and the "
    "measured compute latency"
)


@dataclass(frozen=True)
class Backing:
    """The datum behind one objective: how to reach it, and what it requires."""

    requires: str
    reader: Reader

    def available(self, view: RecordView) -> bool:
        return self.reader(view) is not None

    def extract(self, view: RecordView) -> float:
        value = self.reader(view)
        if value is None:
            raise ValueError(
                f"objective backing is empty on the {view.view_kind} view: "
                f"requires {self.requires}"
            )
        return float(value)


def view_field(name: str, requires: str) -> Backing:
    """A datum the view itself declares (the candidate's static facts)."""

    def read(view: RecordView) -> Optional[float]:
        value = getattr(view, name)
        return None if value is None else float(value)

    return Backing(requires=requires, reader=read)


def layout_field(name: str) -> Backing:
    """A layout-verification statistic — the candidate's stats or the record's mirror."""

    def read(view: RecordView) -> Optional[float]:
        layout = view.layout
        return None if layout is None else float(getattr(layout, name))

    return Backing(requires=_LAYOUT_REQUIRES, reader=read)


def sync_barrier_backing() -> Backing:
    """Host-side segment slots plus the schedule's pass syncs — the candidate census."""

    def read(view: RecordView) -> Optional[float]:
        layout = view.layout
        host_segments = view.host_side_segment_count
        if layout is None or host_segments is None:
            return None
        return float(host_segments + layout.schedule_sync_count)

    return Backing(requires=_SYNC_REQUIRES, reader=read)


def schedule_field(name: str) -> Backing:
    """A census field of the sealed schedule fragment."""

    def read(view: RecordView) -> Optional[float]:
        record = view.record
        return None if record is None else float(getattr(record.schedule, name))

    return Backing(requires=_SCHEDULE_REQUIRES, reader=read)


def reprogramming_bytes_backing() -> Backing:
    """Σ programmed weight bytes over the segments a pass actually reprograms."""

    def read(view: RecordView) -> Optional[float]:
        record = view.record
        if record is None:
            return None
        return float(
            sum(
                segment.params_bytes
                for segment in record.schedule.segments()
                if segment.programming == "reprogram"
            )
        )

    return Backing(requires=_SCHEDULE_REQUIRES, reader=read)


def energy_field(name: str) -> Backing:
    """A measured field of the energy fragment (SANA-FE)."""

    def read(view: RecordView) -> Optional[float]:
        record = view.record
        if record is None or record.energy is None:
            return None
        return float(getattr(record.energy, name))

    return Backing(requires=_ENERGY_REQUIRES, reader=read)


def deployed_accuracy_backing() -> Backing:
    """The record's deployed accuracy read."""

    def read(view: RecordView) -> Optional[float]:
        record = view.record
        return None if record is None else float(record.accuracy.deployed.metric)

    return Backing(
        requires="the sealed record's accuracy fragment (a deployed read)",
        reader=read,
    )


def latency_steps_backing() -> Backing:
    """Executed timesteps, from the MEASURED per-segment timing census."""

    def read(view: RecordView) -> Optional[float]:
        record = view.record
        if record is None or not record.timing.per_segment:
            return None
        return float(record.timing.latency.compute_steps)

    return Backing(requires=_TIMING_REQUIRES, reader=read)


def host_op_wall_backing() -> Backing:
    """The measured whole-run host-ComputeOp wall, when any op was timed."""

    def read(view: RecordView) -> Optional[float]:
        record = view.record
        if record is None or record.timing.latency.host_ops_s is None:
            return None
        return float(record.timing.latency.host_ops_s)

    return Backing(
        requires="a timed host ComputeOp wall (timing.latency.host_ops_s)",
        reader=read,
    )


def noc_field(name: str) -> Backing:
    """A SANA-FE NoC packet census field."""

    def read(view: RecordView) -> Optional[float]:
        record = view.record
        if record is None or record.traffic is None or record.traffic.noc is None:
            return None
        return float(getattr(record.traffic.noc, name))

    return Backing(requires=_NOC_REQUIRES, reader=read)


def cost_term(
    group: str, term_name: str, *, requires: str = _COST_REQUIRES
) -> Backing:
    """A term of a cost report — imported, never re-derived here.

    A MISSING term is an answer, not a malformed report: a report can legitimately
    be partial (a candidate carries only the vendor-priced plane; a target's physics
    may refuse an axis by name), and availability==extraction must never raise.
    """

    def read(view: RecordView) -> Optional[float]:
        report = view.cost_report()
        if report is None:
            return None
        terms: Sequence[CostTerm] = getattr(report, group)
        term = find_term_or_none(terms, term_name)
        return None if term is None else float(term.value)

    return Backing(requires=requires, reader=read)


def priced_term(group: str, term_name: str) -> Backing:
    """A vendor-priced term: the same reader, requiring declared physics."""
    return cost_term(group, term_name, requires=_PHYSICS_REQUIRES)

