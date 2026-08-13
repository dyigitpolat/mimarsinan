"""The two record views: a sealed deployment record, and a search candidate.

Same shape at two completenesses (schema doc §3): a search candidate populates
the static facts its layout hook computes, a sealed run populates every
fragment. Neither view invents a datum it does not hold — an absent fragment
reads ``None`` and the objectives over it are simply unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

from mimarsinan.deployment_record.cost.absolute import (
    absolute_pricing_for_record,
    candidate_cost_report,
    report_with_absolute_terms,
)
from mimarsinan.deployment_record.cost.model import DeploymentCostModel
from mimarsinan.deployment_record.cost.terms import DeploymentCostReport
from mimarsinan.deployment_record.objectives.spec import LayoutStatsView, RecordView
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.quantities import (
    CandidateQuantityContext,
    Quantities,
    from_candidate,
)
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
    """A search candidate's static facts — exactly what the joint layout hook computes.

    ``physics`` and ``quantity_context`` are stored FRAGMENTS; the quantities they
    price are DERIVED from them plus the layout, so dropping either fragment in a
    probe answers honestly (a pre-baked quantity set would let an energy-only
    objective set skip the mapping and then fail at extraction).
    """

    layout: Optional[LayoutStatsView]
    chip_param_capacity: Optional[float]
    total_params: Optional[float]
    host_side_segment_count: Optional[int]
    estimated_accuracy: Optional[float] = None
    physics: Optional[PlatformPhysics] = None
    quantity_context: Optional[CandidateQuantityContext] = None
    _priced: Dict[str, Optional[DeploymentCostReport]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def view_kind(self) -> str:
        return "candidate_static"

    @property
    def record(self) -> Optional[DeploymentRecord]:
        return None

    @property
    def quantities(self) -> Quantities:
        """What this candidate's shape and declarations can answer."""
        return from_candidate(
            layout=self.layout,
            chip_param_capacity=self.chip_param_capacity,
            total_params=self.total_params,
            host_side_segment_count=self.host_side_segment_count,
            context=self.quantity_context or CandidateQuantityContext(),
        )

    def cost_report(self) -> Optional[DeploymentCostReport]:
        """The vendor-priced report, or None when this run declared no physics."""
        if "report" not in self._priced:
            self._priced["report"] = (
                candidate_cost_report(self.quantities, self.physics)
                if self.physics is not None
                else None
            )
        return self._priced["report"]


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
        """The measured cost surface, plus the vendor-priced plane when declared.

        Either half may be absent: a run without SANA-FE has no measured plane, a
        run without a profile has no priced one, and a run with neither reports no
        report at all.
        """
        if "report" not in self._evaluated:
            self._evaluated["report"] = self._build_report()
        return self._evaluated["report"]

    def _build_report(self) -> Optional[DeploymentCostReport]:
        measured = self.cost_model.evaluate(self.record) if self.costable() else None
        pricing = absolute_pricing_for_record(self.record)
        if pricing is None:
            return measured
        base = measured or DeploymentCostReport(
            segments=(), energy=(), latency=(), area=(), throughput=(), notes=()
        )
        return report_with_absolute_terms(base, pricing)


if TYPE_CHECKING:
    # Both views satisfy the objective contract — checked statically, not by hope.
    _CANDIDATE_IS_A_VIEW: type[RecordView] = CandidateStaticView
    _RECORD_IS_A_VIEW: type[RecordView] = DeploymentRecordView
