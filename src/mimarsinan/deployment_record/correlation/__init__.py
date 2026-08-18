"""Silicon correlation: do the declared constants reproduce the chip's own papers?"""

from mimarsinan.deployment_record.correlation.case import (
    PublishedValue,
    ReferenceCase,
)
from mimarsinan.deployment_record.correlation.library import (
    available_cases,
    correlate_all,
    get_case,
)
from mimarsinan.deployment_record.correlation.report import (
    correlation_payload,
    correlation_payload_json,
    render_correlation,
)
from mimarsinan.deployment_record.correlation.run import (
    AxisCorrelation,
    CaseCorrelation,
    correlate,
    worst_error,
)

__all__ = [
    "AxisCorrelation",
    "CaseCorrelation",
    "PublishedValue",
    "ReferenceCase",
    "available_cases",
    "correlate",
    "correlate_all",
    "correlation_payload",
    "correlation_payload_json",
    "get_case",
    "render_correlation",
    "worst_error",
]
