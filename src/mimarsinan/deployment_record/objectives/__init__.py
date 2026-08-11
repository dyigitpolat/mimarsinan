"""Objectives registry v2: typed, directed, provenance-carrying axes over the record."""

from mimarsinan.deployment_record.objectives.catalog import (
    ACCURACY_OBJECTIVE_KEY,
    OBJECTIVES,
    build_catalog,
    objective,
)
from mimarsinan.deployment_record.objectives.extractors import Backing
from mimarsinan.deployment_record.objectives.registry import ObjectiveRegistry
from mimarsinan.deployment_record.objectives.spec import (
    Direction,
    LayoutStatsView,
    ObjectiveProvenance,
    ObjectiveSpecV2,
    RecordView,
)
from mimarsinan.deployment_record.objectives.views import (
    SEARCH_MODES,
    CandidateStaticView,
    DeploymentRecordView,
    candidate_capability_probe,
    chip_param_capacity,
    declared_core_capacity,
    mode_trains_accuracy,
)

__all__ = [
    "ACCURACY_OBJECTIVE_KEY",
    "OBJECTIVES",
    "SEARCH_MODES",
    "Backing",
    "CandidateStaticView",
    "DeploymentRecordView",
    "Direction",
    "LayoutStatsView",
    "ObjectiveProvenance",
    "ObjectiveRegistry",
    "ObjectiveSpecV2",
    "RecordView",
    "build_catalog",
    "candidate_capability_probe",
    "chip_param_capacity",
    "declared_core_capacity",
    "mode_trains_accuracy",
    "objective",
]
