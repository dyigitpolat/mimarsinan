"""Lifetime and K* of an executed program on the write law's own verdicts: K* per endurance corner is the smallest
priced admitting batch, inadmissible one below it; operand writes per inference take one verdict at every batch.
[U-C C-1] The endurance is read in writes: the band's cycles over the technology's pulses per write."""

from __future__ import annotations

import math

from mimarsinan.mapping.support.core_slots import SlotWrites
from mimarsinan.deployment_record.build.write_census import WriteCensus
from mimarsinan.deployment_record.cost.write_lifetime import endurance_in_writes
from mimarsinan.deployment_record.design_point.endurance import CORNERS, k_star, lifetime_at
from mimarsinan.deployment_record.design_point.timeline import ProgramWrites, program_writes
from mimarsinan.deployment_record.quantities.write_law import write_policy_of
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from . import programs as P

GRID = (1, 16, 64)


def _writes(count):
    platform = P.platform(count)
    return program_writes(P.built(count), platform, weight_bits=platform["weight_bits"])


def test_k_star_per_corner_is_the_priced_boundary():
    writes, physics, platform = _writes(3), P.neurram(), P.platform(3)
    reading = k_star(writes, physics, platform, GRID)
    assert reading["checked"] and reading["no_K_admits"] is None
    by_k = {v["K"]: v for v in reading["verdicts_by_K"]}
    for corner, value in reading["per_corner"].items():
        assert value is not None and by_k[value]["per_corner"][corner] == "admitted"
        if value > 1:
            assert by_k[value - 1]["per_corner"][corner] == "inadmissible"
    low, nominal, high = (reading["per_corner"][c] for c in CORNERS)
    assert low >= nominal >= high


def test_k_star_equals_the_closed_form_of_the_static_wear():
    writes, physics, platform = _writes(3), P.neurram(), P.platform(3)
    most = writes.census.most_written_core
    reading = k_star(writes, physics, platform, GRID)
    endurance = endurance_in_writes(physics)
    assert endurance is not None
    for corner, cycles in zip(CORNERS, (endurance.low, endurance.nominal, endurance.high)):
        closed = most * P.RATE_HZ * 3600.0 * P.FLOOR_H / cycles
        assert reading["per_corner"][corner] == max(1, math.ceil(closed - 1e-9))


def test_lifetime_at_the_declared_batch_reads_the_write_law():
    writes, physics, platform = _writes(3), P.neurram(), P.platform(3)
    life = lifetime_at(writes, physics, platform, 1)
    assert life["at_K"] == 1 and life["verdict"] in ("inadmissible", "straddles", "admitted")
    assert life["hours"][0] <= life["hours"][1] <= life["hours"][2]
    assert life["endurance"]["evidence_kind"] == physics.constants["endurance_cycles"].evidence_kind


def _operand_census():
    passes = (PassCensus(position=0, segment_index=0, resident=True, cores=2, resident_cores=2,
                         written_cells_backing_store=0, written_cells_producer=4096, payload_bytes=0,
                         compute_steps=0),)
    return WriteCensus(census=PassPeriodCensus(passes=passes, label="test: one operand pass"), written_extents=None,
                       compute_sealed=False, slot_writes=(SlotWrites(core_type=(256, 32), writes=(0, 0)),))


def test_operand_writes_every_inference_take_one_verdict_at_every_batch_on_resistive_endurance():
    """No batch amortizes a producer write: a corner whose endurance in writes clears the floor at one write per
    inference admits from K = 1, every other corner at no K (U-C C-1: the pulse-counted resistive band straddles)."""
    platform = P.platform(2)
    writes = ProgramWrites(write_policy_of(platform), _operand_census())
    physics = P.neurram()
    reading = k_star(writes, physics, platform, GRID)
    endurance = endurance_in_writes(physics)
    assert endurance is not None
    for corner, cycles in zip(CORNERS, (endurance.low, endurance.nominal, endurance.high)):
        admits = cycles / (P.RATE_HZ * 3600.0) >= P.FLOOR_H
        assert reading["per_corner"][corner] == (1 if admits else None), corner
    assert "operand cells are written every inference" in reading["no_K_admits"]
    assert len({(v["verdict"], tuple(sorted(v["per_corner"].items()))) for v in reading["verdicts_by_K"]}) == 1
