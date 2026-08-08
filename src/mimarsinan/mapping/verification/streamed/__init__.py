"""Streamed span-topology reporting [plan §9]."""

from mimarsinan.mapping.verification.streamed.streamability import (
    NotStreamableError,
    StreamedSpanReport,
    streamed_span_report_ir,
    streamed_span_report_model,
)

__all__ = [
    "NotStreamableError",
    "StreamedSpanReport",
    "streamed_span_report_ir",
    "streamed_span_report_model",
]
