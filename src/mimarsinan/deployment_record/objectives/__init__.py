"""Objectives registry v2: typed, directed, provenance-carrying axes over the record."""

from mimarsinan.deployment_record.objectives.catalog import (
    ACCURACY_OBJECTIVE_KEY,
    OBJECTIVES,
    build_catalog,
    objective,
)
from mimarsinan.deployment_record.objectives.context import (
    candidate_context_from_platform,
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
from mimarsinan.deployment_record.objectives.probes import (
    CANDIDATE_FRAGMENTS,
    candidate_capability_probe,
    candidate_probe_without,
    full_candidate_probe,
    run_capability_probe,
)
from mimarsinan.deployment_record.objectives.views import (
    SEARCH_MODES,
    CandidateStaticView,
    DeploymentRecordView,
    chip_param_capacity,
    declared_core_capacity,
    mode_trains_accuracy,
)

__all__ = [
    "ACCURACY_OBJECTIVE_KEY",
    "CANDIDATE_FRAGMENTS",
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
    "candidate_context_from_platform",
    "candidate_probe_without",
    "full_candidate_probe",
    "run_capability_probe",
    "chip_param_capacity",
    "declared_core_capacity",
    "mode_trains_accuracy",
    "objective",
]
