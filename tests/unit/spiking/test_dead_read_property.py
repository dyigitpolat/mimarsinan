"""[U-TGAUGE6] The dead-read class by a generator: the unit's generated cascades (``cascade_trees``) with dead
in-segment reads drawn (input columns at zero from a producer other cores still read, a producer read only through a
dead read, a chain of such producers), built through the mapping in the timing family and in the count family under
both LIF semantics. Every tree reads as the deployed executor element by element with each reading core's fold equal
to its sources' placed gauges, or the toolchain declines it by name (never the window refusal: the mapping arms the
relays); any other error fails the test. Each chunk draws the class: dead reads turned off and cores removed.

On U-TGAUGE5's tree the windowed LIF semantics failed with an unnamed error wherever a level stage was left without
outputs. 240 trees per family, in chunks.
"""

from __future__ import annotations

import pytest

import unit.spiking.cascade_trees as ct

TREES = 240
CHUNK = 40
FOLD_TOLERANCE = 1e-5
FAMILIES = [("timing", None), ("count", "streamed_lif"), ("count", "windowed_lif")]


@pytest.mark.parametrize("first", range(0, TREES, CHUNK))
@pytest.mark.parametrize("family,semantics", FAMILIES, ids=["timing", "count_streamed", "count_windowed"])
def test_every_generated_dead_read_reads_as_the_deployed_executor_through_the_mapping(family, semantics, first):
    compared, refused, off, folds, reads, removed = 0, [], [], [], 0, 0
    for seed in range(first, first + CHUNK):
        rec = ct.read(seed, family, semantics=semantics, dead=True)
        if "refused" in rec:
            refused.append(rec)
            continue
        compared += 1
        if rec["off"]:
            off.append(rec)
        if rec["fold_error"] > FOLD_TOLERANCE:
            folds.append(rec)
        reads += rec["window"].get("dead_reads_off", 0)
        removed += rec["window"].get("unread_cores_removed", 0)
    assert not off, off[:3]
    assert not folds, folds[:3]
    assert not [r for r in refused if "UnbalancedFanInError" in r["refused"]], refused[:3]
    assert compared >= CHUNK - 2, refused  # the generator draws constructions the toolchain builds
    assert reads > 0 and removed > 0, (reads, removed)  # the chunk draws the class
