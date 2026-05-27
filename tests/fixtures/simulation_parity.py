"""Shared fakes for Loihi/SANA-FE parity pipeline step tests."""

from __future__ import annotations

from typing import Any


def fake_load_test_sample(sample_index: int = 0):
    import torch

    return torch.zeros(1, 28, 28), 0


def fake_record_hcm_reference(
    pipeline,
    hybrid_mapping,
    sample,
    *,
    sample_index: int = 0,
    device: str | None = None,
    calls: dict | None = None,
):
    import torch
    from mimarsinan.chip_simulation.recording.spike_recorder import RunRecord

    if calls is not None:
        calls.setdefault("hcm_samples", []).append(sample_index)
    ref = RunRecord()
    flow = torch.nn.Identity()
    return flow, ref


def fake_assert_spike_parity_or_raise(ref, actual) -> None:
    return None
