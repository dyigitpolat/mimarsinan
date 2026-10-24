"""The seam of N2b's per-core-type write technology and the closure's sign register (CL-SG) in the period census.

A per-axon core's payload is its cells and its per-row sign register (``payload_sizes.core_payload_bytes``, read by
``period_census`` through ``PassRow.core_bytes``). N2b splits the period census into each write technology's share
and prices each share under its own physics; every share must size its rewritten cores by the same rule, so the
shares sum to the census field by field, sign register bytes included, and each technology commits its own cores'
registers."""

from __future__ import annotations

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.deployment_record.quantities.write_law import write_policy_of
from mimarsinan.deployment_record.quantities.write_technology import split_by_technology

S, T = (64, 64), (16, 16)  # a per-axon type of the chip's physics, a per-synapse type of a named write physics


def _platform():
    return {"cores": [
        {"max_axons": 64, "max_neurons": 64, "count": 1, "write_granularity": "cell",
         "weight_sign_granularity": "per_axon"},
        {"max_axons": 16, "max_neurons": 16, "count": 1, "write_granularity": "cell", "write_physics": "neurram"},
    ], "inference_batch_K": 1}


def _row(segment, cells, signs):
    return PassRow(segment_index=segment, resident=False, core_cells=tuple(cells), marks=frozenset(),
                   core_extents=tuple((c, 1) for c in cells), compute_steps=1, stage_latencies=(0,),
                   core_types=(S, T), core_sign_rows=tuple(signs))


def _passes():
    """Two passes on one core of each type, so every pass rewrites both cores (the program rotates, R3)."""
    return ProgramPasses(rows=(_row(0, (10, 7), (9, 0)), _row(1, (20, 5), (17, 0))))


def test_the_census_counts_the_sign_register_of_the_per_axon_core():
    written = period_census(_passes(), weight_bits=8, capacity={S: 1, T: 1})
    # 10 cells + ceil(9 / 8) = 2 sign bytes, and 7; then 20 + ceil(17 / 8) = 3, and 5
    assert [p.payload_bytes for p in written.census.passes] == [19, 28]


def test_each_write_technology_share_sizes_its_cores_by_the_one_payload_rule():
    passes = _passes()
    written = period_census(passes, weight_bits=8, capacity={S: 1, T: 1})
    split = split_by_technology(passes, written, write_policy_of(_platform()), weight_bits=8)
    assert split.refused is None and len(split.parts) == 2
    by_type = {frozenset(t.core_type for t in part.group): part for part in split.parts}
    assert [p.payload_bytes for p in by_type[frozenset({S})].census.passes] == [12, 23]
    assert [p.payload_bytes for p in by_type[frozenset({T})].census.passes] == [7, 5]
    for position, full in enumerate(written.census.passes):
        assert sum(part.census.passes[position].payload_bytes for part in split.parts) == full.payload_bytes
