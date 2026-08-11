"""The objective contract: one typed, directed, provenance-carrying optimization axis.

An objective is a QUESTION asked of a view over the deployment artifact, not a
number a producer happens to publish. Two things are therefore declared next to
each other and never drift: ``availability`` — is the backing datum populated in
this view — and ``extractor`` — the value, once it is. Reading an unavailable
objective raises; that is the honesty discipline of the schema doc (§3) made
executable, replacing the search surface's silent drop.

``name``/``goal`` are the legacy ``search.results.ObjectiveSpec`` field names,
kept as properties so every optimizer that reads ``spec.name``/``spec.goal``
runs unchanged against a v2 spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol

from mimarsinan.deployment_record.cost.terms import DeploymentCostReport
from mimarsinan.deployment_record.schema import DeploymentRecord
from mimarsinan.deployment_record.schema.serde import require_choice

Direction = Literal["min", "max"]
ObjectiveProvenance = Literal["measured", "modeled", "static", "training_proxy"]

OBJECTIVE_DIRECTIONS = frozenset({"min", "max"})
OBJECTIVE_PROVENANCES = frozenset(
    {"measured", "modeled", "static", "training_proxy"}
)


class LayoutStatsView(Protocol):
    """The layout-stats fields objectives read.

    Satisfied by ``mapping``'s ``LayoutVerificationStats`` (the search seam) and
    by its record mirror ``LayoutStatsRecord`` — one extractor serves both.
    """

    @property
    def mapped_params_pct(self) -> float: ...
    @property
    def total_wasted_axons_pct(self) -> float: ...
    @property
    def total_wasted_neurons_pct(self) -> float: ...
    @property
    def fragmentation_pct(self) -> float: ...
    @property
    def schedule_sync_count(self) -> int: ...


class RecordView(Protocol):
    """What an objective may read; an accessor is ``None`` exactly when its datum is absent."""

    @property
    def view_kind(self) -> str: ...
    @property
    def layout(self) -> Optional[LayoutStatsView]: ...
    @property
    def total_params(self) -> Optional[float]: ...
    @property
    def chip_param_capacity(self) -> Optional[float]: ...
    @property
    def host_side_segment_count(self) -> Optional[int]: ...
    @property
    def estimated_accuracy(self) -> Optional[float]: ...
    @property
    def record(self) -> Optional[DeploymentRecord]: ...

    def cost_report(self) -> Optional[DeploymentCostReport]: ...


@dataclass(frozen=True)
class ObjectiveSpecV2:
    """One optimization axis: what it is, which way is better, and what backs it."""

    key: str
    direction: Direction
    unit: str
    provenance: ObjectiveProvenance
    requires: str
    availability: Callable[[RecordView], bool]
    extractor: Callable[[RecordView], float]
    doc: str

    def __post_init__(self) -> None:
        require_choice("ObjectiveSpecV2", "direction", self.direction, OBJECTIVE_DIRECTIONS)
        require_choice(
            "ObjectiveSpecV2", "provenance", self.provenance, OBJECTIVE_PROVENANCES
        )
        for field_name in ("key", "unit", "requires", "doc"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"ObjectiveSpecV2.{field_name} must be stated, got empty "
                    f"(objective {self.key!r})"
                )

    @property
    def name(self) -> str:
        """The legacy ``ObjectiveSpec.name`` every optimizer reads."""
        return self.key

    @property
    def goal(self) -> Direction:
        """The legacy ``ObjectiveSpec.goal`` every optimizer reads."""
        return self.direction

    def available(self, view: RecordView) -> bool:
        """Is this objective's backing datum populated in *view*?"""
        return bool(self.availability(view))

    def value(self, view: RecordView) -> float:
        """The objective's value, or a loud refusal naming what it requires."""
        if not self.available(view):
            raise ValueError(
                f"objective {self.key!r} is not available on the "
                f"{view.view_kind} view: requires {self.requires}"
            )
        return float(self.extractor(view))
