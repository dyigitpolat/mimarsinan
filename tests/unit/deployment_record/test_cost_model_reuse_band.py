"""Coefficient-application parity: the model's bands ARE ``phase_cost_band``.

The whole point of costing the record with ``weight_reuse_cost_model`` is that
the coefficients are applied by ITS functions, on real record quantities. So
each modeled DMA/sync band here is compared — exactly, no tolerance — against
``phase_cost_band`` called directly on the same quantities, corner by corner:

* the payload energy goes through the per-BYTE channel (the record's
  ``params_bytes`` is already a byte count, so the weight-width coefficient
  must NOT be applied again — pinned by the discriminating test below),
* the sync energy goes through the barrier channel on ``schedule.sync_count``.
"""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.weight_reuse_cost_model import (
    DEFAULT_COEFFICIENT_BAND,
    phase_cost_band,
)
from mimarsinan.deployment_record.cost import DeploymentCostModel, find_term
from unit.deployment_record.record_fixtures import make_full_record

MODEL = DeploymentCostModel()


def byte_channel_mj(payload_bytes: float, corner: str) -> float:
    """``phase_cost_band`` called directly on these bytes, read at ``corner``."""
    band = phase_cost_band(
        reprogram_passes=0,
        reuse_passes=0,
        params_reloaded=0,
        activation_bytes_moved=int(payload_bytes),
    )
    return getattr(band, f"{corner}_mj")


def param_channel_mj(params: float, corner: str) -> float:
    """The WRONG channel for a byte count: it multiplies by bytes_per_param."""
    band = phase_cost_band(
        reprogram_passes=0,
        reuse_passes=0,
        params_reloaded=int(params),
        activation_bytes_moved=0,
    )
    return getattr(band, f"{corner}_mj")


def barrier_channel_mj(barriers: int, corner: str) -> float:
    band = phase_cost_band(
        reprogram_passes=barriers,
        reuse_passes=0,
        params_reloaded=0,
        activation_bytes_moved=0,
    )
    return getattr(band, f"{corner}_mj")


@pytest.fixture
def record():
    return make_full_record()


def band_corners(band):
    return {"low": band.low, "nominal": band.nominal, "high": band.high}


def test_segment_payload_energy_equals_phase_cost_band_on_the_same_bytes(record):
    for cost in MODEL.segment_initialization(record):
        payload = band_corners(cost.term("payload_bytes").band)
        energy = band_corners(cost.term("payload_energy_mj").band)
        for corner, bytes_at_corner in payload.items():
            assert energy[corner] == byte_channel_mj(bytes_at_corner, corner), (
                f"{cost.programming} segment, {corner} corner"
            )


def test_total_programming_energy_equals_the_direct_call_per_segment(record):
    modeled = band_corners(
        find_term(MODEL.energy(record), "modeled_programming_mj").band
    )
    segments = MODEL.segment_initialization(record)
    for corner, value in modeled.items():
        expected = sum(
            byte_channel_mj(band_corners(cost.term("payload_bytes").band)[corner], corner)
            for cost in segments
        )
        assert value == expected, corner


def test_sync_energy_equals_phase_cost_band_on_the_sync_census(record):
    sync = band_corners(find_term(MODEL.energy(record), "modeled_sync_mj").band)
    for corner, value in sync.items():
        assert value == barrier_channel_mj(record.schedule.sync_count, corner)


def test_the_weight_width_is_not_applied_to_an_already_byte_sized_payload(record):
    """params_bytes is bytes: the param channel would re-apply bytes_per_param."""
    cost = MODEL.segment_initialization(record)[0]
    payload = band_corners(cost.term("payload_bytes").band)
    energy = band_corners(cost.term("payload_energy_mj").band)
    for corner, bytes_at_corner in payload.items():
        width = getattr(DEFAULT_COEFFICIENT_BAND, corner).bytes_per_param
        wrong = param_channel_mj(bytes_at_corner, corner)
        assert wrong == pytest.approx(energy[corner] * width, rel=1e-12)
        if width != 1.0:
            assert energy[corner] != wrong, corner


def test_a_resident_segment_moves_no_bytes_through_the_dma_channel(record):
    resident = MODEL.segment_initialization(record)[1]
    assert resident.programming == "resident"
    energy = band_corners(resident.term("payload_energy_mj").band)
    for corner, value in energy.items():
        assert value == byte_channel_mj(0.0, corner) == 0.0
