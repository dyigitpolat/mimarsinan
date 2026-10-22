"""The absolute pricer: vendor physics × recorded quantities → attributable costs."""

from mimarsinan.deployment_record.cost.absolute.context import (
    AbsolutePricing,
    AdmissibilityVerdict,
    PricingContext,
    PricingRefusal,
)
from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.validity import CENSUS_PROVED, UNJUDGED, validity_refusals
from mimarsinan.deployment_record.cost.absolute.writes import (
    WriteInputs,
    price_write_terms,
)
from mimarsinan.deployment_record.cost.absolute.report import (
    ABSOLUTE_TERM_NAMES,
    DEPLOYMENT_COST_REPORT_FILENAME,
    absolute_pricing_for_record,
    candidate_cost_report,
    declared_physics_of,
    emit_physics_report,
    report_with_absolute_terms,
)

__all__ = [
    "CENSUS_PROVED",
    "UNJUDGED",
    "ABSOLUTE_TERM_NAMES",
    "AbsolutePricing",
    "AdmissibilityVerdict",
    "DEPLOYMENT_COST_REPORT_FILENAME",
    "PricingContext",
    "PricingRefusal",
    "absolute_pricing_for_record",
    "candidate_cost_report",
    "declared_physics_of",
    "emit_physics_report",
    "price_absolute",
    "price_write_terms",
    "report_with_absolute_terms",
    "validity_refusals",
    "WriteInputs",
]
