"""Cost surfaces over the deployment record: legacy continuity + the cost model."""

from mimarsinan.deployment_record.cost.coefficients import (
    BYTES_PER_CONNECTIVITY_ENTRY,
    CORE_INIT,
    DMA_COEFFICIENT_BAND,
    PROGRAMMING_BANDWIDTH_BYTES_PER_S,
    SYNC_BARRIER_S,
    CoreInitCoefficients,
)
from mimarsinan.deployment_record.cost.absolute import (
    ABSOLUTE_TERM_NAMES,
    AbsolutePricing,
    absolute_pricing_for_record,
    candidate_cost_report,
    emit_physics_report,
    price_absolute,
    report_with_absolute_terms,
)
from mimarsinan.deployment_record.cost.legacy_projection import (
    cost_record_from_deployment_record,
)
from mimarsinan.deployment_record.cost.model import DeploymentCostModel
from mimarsinan.deployment_record.cost.terms import (
    COST_REPORT_FORMAT_VERSION,
    CostTerm,
    DeploymentCostReport,
    SegmentInitCost,
    find_term,
)

__all__ = [
    "ABSOLUTE_TERM_NAMES",
    "BYTES_PER_CONNECTIVITY_ENTRY",
    "CORE_INIT",
    "COST_REPORT_FORMAT_VERSION",
    "AbsolutePricing",
    "CostTerm",
    "absolute_pricing_for_record",
    "candidate_cost_report",
    "emit_physics_report",
    "price_absolute",
    "report_with_absolute_terms",
    "CoreInitCoefficients",
    "DMA_COEFFICIENT_BAND",
    "DeploymentCostModel",
    "DeploymentCostReport",
    "PROGRAMMING_BANDWIDTH_BYTES_PER_S",
    "SYNC_BARRIER_S",
    "SegmentInitCost",
    "cost_record_from_deployment_record",
    "find_term",
]
