"""Deployment-faithfulness certificates (spike-count observable SSOT)."""

from mimarsinan.certification.spike_certificate import (
    SpikeCountCertificate,
    certify_spike_counts,
)

__all__ = ["SpikeCountCertificate", "certify_spike_counts"]
