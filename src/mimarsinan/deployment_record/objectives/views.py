"""The two record views: a sealed deployment record, and a search candidate.

Same shape at two completenesses (schema doc §3): a search candidate populates
the static facts its layout hook computes, a sealed run populates every
fragment. Neither view invents a datum it does not hold — an absent fragment
reads ``None`` and the objectives over it are simply unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

from mimarsinan.deployment_record.cost.model import DeploymentCostModel
from mimarsinan.deployment_record.cost.terms import DeploymentCostReport
from mimarsinan.deployment_record.objectives.spec import LayoutStatsView, RecordView
from mimarsinan.deployment_record.schema import DeploymentRecord

CORE_CAPACITY_KEYS = ("max_axons", "max_neurons", "count")

# Searchable modes; ``fixed`` runs no search, so it declares no catalog.
HARDWARE_SEARCH_MODE = "hardware"
SEARCH_MODES = ("model", "hardware", "joint")


def chip_param_capacity(cores: Sequence[Mapping[str, Any]]) -> float:
    """Σ ``max_axons × max_neurons × count`` — the platform's declared cell capacity."""
    return float(
        sum(
            int(ct["max_axons"]) * int(ct["max_neurons"]) * int(ct["count"])
            for ct in cores
        )
    )


def declared_core_capacity(platform: Mapping[str, Any]) -> Optional[float]:
    """The capacity a resolved platform DECLARES, or ``None`` when it declares none."""
    cores = platform.get("cores")
    if not isinstance(cores, SequenceABC) or isinstance(cores, (str, bytes)) or not cores:
        return None
    for core_type in cores:
        if not isinstance(core_type, MappingABC):
            return None
        if any(key not in core_type for key in CORE_CAPACITY_KEYS):
            return None
    return chip_param_capacity(cores)


def mode_trains_accuracy(search_mode: str) -> bool:
    """Hardware-only search trains nothing, so its candidates carry no accuracy."""
    return search_mode != HARDWARE_SEARCH_MODE


@dataclass(frozen=True)
class CandidateStaticView:
    """A search candidate's static facts — exactly what the joint layout hook computes."""

    layout: Optional[LayoutStatsView]
    chip_param_capacity: Optional[float]
    total_params: Optional[float]
    host_side_segment_count: Optional[int]
    estimated_accuracy: Optional[float] = None

    @property
    def view_kind(self) -> str:
        return "candidate_static"

    @property
    def record(self) -> Optional[DeploymentRecord]:
        return None

    def cost_report(self) -> Optional[DeploymentCostReport]:
        return None


@dataclass(frozen=True)
class DeploymentRecordView:
    """A sealed record as an objective view; the cost surface is evaluated once."""

    record: DeploymentRecord
    cost_model: DeploymentCostModel = field(default_factory=DeploymentCostModel)
    _evaluated: Dict[str, Optional[DeploymentCostReport]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def view_kind(self) -> str:
        return "deployment_record"

    @property
    def layout(self) -> Optional[LayoutStatsView]:
        return self.record.utilization.layout

    @property
    def total_params(self) -> Optional[float]:
        """No fragment carries the model's parameter census; the candidate view does."""
        return None

    @property
    def chip_param_capacity(self) -> Optional[float]:
        return declared_core_capacity(self.record.identity.platform)

    @property
    def host_side_segment_count(self) -> Optional[int]:
        """No fragment carries the host-slot census; the candidate view does."""
        return None

    @property
    def estimated_accuracy(self) -> Optional[float]:
        """A search-time proxy has no place in a sealed record; ``deployed_accuracy`` does."""
        return None

    def costable(self) -> bool:
        """The cost model's own preconditions: measured energy and compute latency."""
        return (
            self.record.energy is not None
            and self.record.timing.latency.compute_sim_time_s is not None
        )

    def cost_report(self) -> Optional[DeploymentCostReport]:
        if "report" not in self._evaluated:
            self._evaluated["report"] = (
                self.cost_model.evaluate(self.record) if self.costable() else None
            )
        return self._evaluated["report"]


@dataclass(frozen=True)
class _ProbeLayout:
    """A layout-stats stand-in whose only claim is that the fields EXIST."""

    mapped_params_pct: float = 0.0
    total_wasted_axons_pct: float = 0.0
    total_wasted_neurons_pct: float = 0.0
    fragmentation_pct: float = 0.0
    schedule_sync_count: int = 0


# The static facts a candidate view can hold; each one is a separate question
# ("does this candidate carry a layout?"), so an objective's need for a fact is
# answered by asking the registry, never by a hand-kept list of objective names.
CANDIDATE_FRAGMENTS: Tuple[str, ...] = (
    "layout",
    "chip_param_capacity",
    "total_params",
    "host_side_segment_count",
    "estimated_accuracy",
)


def _full_candidate_probe() -> CandidateStaticView:
    """Every candidate fragment populated; the values are placeholders."""
    return CandidateStaticView(
        layout=_ProbeLayout(),
        chip_param_capacity=0.0,
        total_params=0.0,
        host_side_segment_count=0,
        estimated_accuracy=0.0,
    )


def candidate_capability_probe(search_mode: str) -> CandidateStaticView:
    """A maximally populated candidate view: what a candidate CAN carry in this mode.

    A capability question, not data — the zeros are placeholders whose only
    meaning is "this datum exists in this mode".
    """
    probe = _full_candidate_probe()
    if mode_trains_accuracy(search_mode):
        return probe
    return replace(probe, estimated_accuracy=None)


def candidate_probe_without(fragment: str) -> CandidateStaticView:
    """A fully populated candidate view MINUS one fragment.

    Asking which objectives go unavailable on it is how a caller learns whether
    a fragment is worth computing — the registry answers, so a new objective
    classifies itself.
    """
    if fragment not in CANDIDATE_FRAGMENTS:
        raise ValueError(
            f"unknown candidate fragment {fragment!r}; a candidate view carries "
            f"{list(CANDIDATE_FRAGMENTS)}"
        )
    return replace(_full_candidate_probe(), **{fragment: None})


if TYPE_CHECKING:
    # Both views satisfy the objective contract — checked statically, not by hope.
    _CANDIDATE_IS_A_VIEW: type[RecordView] = CandidateStaticView
    _RECORD_IS_A_VIEW: type[RecordView] = DeploymentRecordView
    _PROBE_LAYOUT_IS_A_LAYOUT: type[LayoutStatsView] = _ProbeLayout
