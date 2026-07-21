"""[calculus §17/PR45] typed certificates over backend RunRecords (Edge B)."""

from __future__ import annotations

import torch

from mimarsinan.certification.spike_certificate import (
    SpikeCountCertificate,
    certify_spike_counts,
)


def _record_counts(record) -> dict:
    out: dict = {}
    for si, seg in record.segments.items():
        out[("seg", int(si), "out")] = torch.as_tensor(
            seg.seg_output_spike_count, dtype=torch.float64
        ).reshape(1, -1)
        for core in getattr(seg, "cores", []):
            key = ("core", int(si), int(core.core_index))
            out[key + ("in",)] = torch.as_tensor(
                core.input_spike_count, dtype=torch.float64
            ).reshape(1, -1)
            out[key + ("out",)] = torch.as_tensor(
                core.output_spike_count, dtype=torch.float64
            ).reshape(1, -1)
    return out


def certify_run_records(ref, actual, *, backend: str) -> SpikeCountCertificate:
    """Certificate over two backend ``RunRecord``s: per-core input/output counts
    plus segment outputs, at the backend class's count tolerance. Both records
    describe the SAME hard-core program, so keys must match exactly (Edge B —
    chip-math executors)."""
    return certify_spike_counts(
        lambda _b: _record_counts(ref),
        lambda _b: _record_counts(actual),
        [torch.zeros(1, 1)],
        backend=backend,
    )
